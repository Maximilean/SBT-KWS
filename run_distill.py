import hydra
import omegaconf
from pytorch_lightning import seed_everything

from src.module import KWSDistillation
from utils import omegaconf_extension


@omegaconf_extension
@hydra.main(version_base="1.2", config_path="conf", config_name="distill.yaml")
def main(conf: omegaconf.DictConfig) -> None:
    seed_everything(314, workers=True)
    module = KWSDistillation(
        conf=conf,
        teacher_ckpt=conf.distill.teacher_ckpt,
        temperature=conf.distill.temperature,
        alpha=conf.distill.alpha,
        teacher_ckpt_2=conf.distill.get("teacher_ckpt_2"),
    )
    logger = hydra.utils.instantiate(conf.logger)
    trainer = hydra.utils.instantiate(conf.trainer, logger=logger)
    trainer.fit(module)


if __name__ == "__main__":
    main()
