# Keyword Spotting - SBT HW

[Kaggle Competition](https://www.kaggle.com/competitions/keyword-spotting-mipt-2025)

constraints:
* $\le 10^4$ params
* $\le 10^6$ multiply-accumulate operations per 1 second

## Training

```bash
# monitor learning curves in tensorboard
tensorboard --logdir ./lightning_logs
```

### Conv1d (baseline)

```bash
# train
python run.py \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest>

# submit
python submit.py ++init_weights=<path_to_model>
```

### DS-CNN

```bash
# train
python run.py --config-name dscnn.yaml \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest>

# submit
python submit.py --config-name dscnn.yaml ++init_weights=<path_to_model>
```

### BC-ResNet

```bash
# train
python run.py --config-name bcresnet.yaml \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest>

# submit
python submit.py --config-name bcresnet.yaml ++init_weights=<path_to_model>
```

### Distillation (DS-CNN student)

```bash
# train
python run_distill.py \
  ++train_dataloader.dataset.manifest_path=<train_manifest> \
  ++val_dataloader.dataset.manifest_path=<val_manifest> \
  ++predict_dataloader.dataset.manifest_path=<test_manifest> \
  ++distill.teacher_ckpt=<teacher_ckpt>

# submit (student shares the DS-CNN architecture)
python submit.py --config-name dscnn.yaml ++init_weights=<path_to_distilled_model>
```