# Remote Sensing GCD Baselines

This folder contains adapted official baseline repositories for AID and NWPU-RESISC45
experiments. Slot-GCD is not included because no public official repository was found.

## Adapted Baselines

All adapted baselines support:

- `--dataset_name aid|nwpu`
- `--split_type random|confusable`
- `--class_split_seed 0`
- `--prop_train_labels 0.5`
- `DATASET_DIR=/path/to/data`

Expected data roots:

- `${DATASET_DIR}/AID`
- `${DATASET_DIR}/NWPU-RESISC45`

The remote-sensing protocol is:

- AID: 15 old / 15 novel classes.
- NWPU-RESISC45: 22 old / 23 novel classes.
- Images are split per class into 70% train / 30% test.
- Old-class train images: 50% labelled.
- Remaining old-class train images plus all novel-class train images: unlabelled.
- Test/evaluation uses the held-out 30% test images.
- Metrics use the standard `v2` evaluation where Hungarian matching is computed once on all unlabelled samples before reporting All/Old/New.

## Entrypoints

- GCD: `baselines/GCD/methods/contrastive_training/contrastive_training.py`
- SimGCD: `baselines/SimGCD/train.py`
- SPTNet: `baselines/SPTNet/train_spt.py`
- AptGCD: `baselines/AptGCD/torch/train.py`
- SelEx: `baselines/SelEx/methods/contrastive_training/contrastive_training.py`
- CMS: `baselines/CMS/methods/contrastive_meanshift_training.py`

## Smoke Test

Expected protocol sizes are:

- AID random: `15/15`, `1767/5233/3000` labelled/unlabelled/test.
- AID confusable: `15/15`, `1844/5156/3000` labelled/unlabelled/test.
- NWPU random/confusable: `22/23`, `5390/16660/9450` labelled/unlabelled/test.

## Known Runtime Requirements

Some official training scripts require extra dependencies such as `tensorboard`.
Several methods also require a valid DINO ViT-B/16 pretrained checkpoint path before
full training can start. Dataset loading does not require these checkpoints.
