from typing import Optional, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchmetrics import Accuracy

# Workaround for torch > 2.6
import pytorch_lightning.core.saving as _pl_saving
_pl_saving.pl_load = lambda path, map_location=None: torch.load(  # type: ignore[assignment]
    path, map_location=map_location, weights_only=False
)


class KWS(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf

        self.model = hydra.utils.instantiate(conf.model)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )

        self.loss = hydra.utils.instantiate(conf.loss)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(inputs)
        preds = logits.argmax(1)
        return logits, preds

    def on_train_start(self):

        # Changed for augmentations compatibility
        features_params = next(
            t for t in self.conf.train_dataloader.dataset.transforms
            if hasattr(t, "n_mels")
        )

        sample_inputs = torch.randn(
            1,
            features_params.n_mels,
            features_params.sample_rate // features_params.hop_length + 1,
            device=self.device,
        )
        macs, params = thop.profile(
            self.model,
            inputs=(sample_inputs,),
        )
        self.log("MACs", macs)
        self.log("Params", params)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logits, preds = self.forward(inputs)

        loss = self.loss(logits, labels)

        log = {
            "train/loss": loss,
            "lr": self.optimizers().param_groups[0]["lr"],
            "train/accuracy": self.train_acc(preds, labels),
        }

        self.log_dict(log, on_step=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        logits, preds = self.forward(inputs)

        loss = self.loss(logits, labels)
        self.valid_acc.update(preds, labels)

        return {"loss": loss}

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        ids, inputs, _ = batch

        _, preds = self.forward(inputs)

        return ids, preds

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.valid_acc.compute())
        self.valid_acc.reset()

    def train_dataloader(self):
        return hydra.utils.instantiate(self.conf.train_dataloader)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.conf.val_dataloader)

    def predict_dataloader(self):
        return hydra.utils.instantiate(self.conf.predict_dataloader)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.conf.optim,
            params=self.model.parameters(),
        )
        result = {"optimizer": optimizer}
        if hasattr(self.conf, "scheduler"):
            warmup_steps = getattr(self.conf, "warmup_steps", 0)
            if warmup_steps > 0:
                cosine_steps = self.trainer.max_steps - warmup_steps
                main_scheduler = hydra.utils.instantiate(
                    self.conf.scheduler,
                    optimizer=optimizer,
                    T_max=cosine_steps,
                )
                warmup = LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, main_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = hydra.utils.instantiate(
                    self.conf.scheduler,
                    optimizer=optimizer,
                    T_max=self.trainer.max_steps,
                )
            result["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return result


class KWSDistillation(pl.LightningModule):
    def __init__(
        self,
        conf,
        teacher_ckpt: str,
        temperature: float = 4.0,
        alpha: float = 0.1,
        teacher_ckpt_2: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf
        self.T = temperature
        self.alpha = alpha

        # Student model
        self.student = hydra.utils.instantiate(conf.model)

        # Teacher (frozen)
        self.teacher = self._load_teacher(teacher_ckpt)
        self.teacher_2 = self._load_teacher(teacher_ckpt_2) if teacher_ckpt_2 else None

        self.hard_loss = hydra.utils.instantiate(conf.distill.loss)

        n_cls = conf.model.n_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=n_cls, top_k=1)
        self.valid_acc = Accuracy(task="multiclass", num_classes=n_cls, top_k=1)

    @staticmethod
    def _load_teacher(ckpt_path: str) -> nn.Module:
        teacher = KWS.load_from_checkpoint(ckpt_path, strict=False)
        model = teacher.model
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        if self.teacher_2 is not None:
            self.teacher_2.eval()
        return self

    @torch.no_grad()
    def _teacher_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.teacher(inputs)
        if self.teacher_2 is not None:
            logits = (logits + self.teacher_2(inputs)) * 0.5
        return logits

    def _kd_loss(
        self,
        s_logits: torch.Tensor,
        t_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        T = self.T
        soft_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        hard_loss = self.hard_loss(s_logits, labels)
        return self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.student(inputs)
        return logits, logits.argmax(1)

    def on_train_start(self):
        features_params = next(
            t for t in self.conf.train_dataloader.dataset.transforms
            if hasattr(t, "n_mels")
        )
        sample = torch.randn(
            1,
            features_params.n_mels,
            features_params.sample_rate // features_params.hop_length + 1,
            device=self.device,
        )
        macs, params = thop.profile(self.student, inputs=(sample,))
        self.log("student_MACs", macs)
        self.log("student_Params", params)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        t_logits = self._teacher_logits(inputs)
        s_logits, preds = self.forward(inputs)
        loss = self._kd_loss(s_logits, t_logits, labels)

        self.log_dict(
            {
                "train/loss": loss,
                "lr": self.optimizers().param_groups[0]["lr"],
                "train/accuracy": self.train_acc(preds, labels),
            },
            on_step=True,
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        self.train_acc.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch
        _, preds = self.forward(inputs)
        self.valid_acc.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log("val/accuracy", self.valid_acc.compute())
        self.valid_acc.reset()

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        ids, inputs, _ = batch
        _, preds = self.forward(inputs)
        return ids, preds

    def train_dataloader(self):
        return hydra.utils.instantiate(self.conf.train_dataloader)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.conf.val_dataloader)

    def predict_dataloader(self):
        return hydra.utils.instantiate(self.conf.predict_dataloader)

    def configure_optimizers(self):
        # Only student parameters optimized — teacher is frozen
        optimizer = hydra.utils.instantiate(
            self.conf.optim,
            params=self.student.parameters(),
        )
        result = {"optimizer": optimizer}
        if hasattr(self.conf, "scheduler"):
            warmup_steps = getattr(self.conf, "warmup_steps", 0)
            if warmup_steps > 0:
                cosine_steps = self.trainer.max_steps - warmup_steps
                main_scheduler = hydra.utils.instantiate(
                    self.conf.scheduler,
                    optimizer=optimizer,
                    T_max=cosine_steps,
                )
                warmup = LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, main_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = hydra.utils.instantiate(
                    self.conf.scheduler,
                    optimizer=optimizer,
                    T_max=self.trainer.max_steps,
                )
            result["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return result
