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
- Old classes: 50% labelled, remaining 50% unlabelled.
- Novel classes: all unlabelled.
- Metrics use the standard `v2` evaluation where Hungarian matching is computed once on all unlabelled samples before reporting All/Old/New.

## Entrypoints

- GCD: `baselines/GCD/methods/contrastive_training/contrastive_training.py`
- SimGCD: `baselines/SimGCD/train.py`
- SPTNet: `baselines/SPTNet/train_spt.py`
- AptGCD: `baselines/AptGCD/torch/train.py`
- SelEx: `baselines/SelEx/methods/contrastive_training/contrastive_training.py`
- CMS: `baselines/CMS/methods/contrastive_meanshift_training.py`

## Smoke Test

Dataset protocol smoke tests passed locally and on the remote 5090 host for:

- AID random: `15/15`, `2525/7475/10000` labelled/unlabelled/test.
- AID confusable: `15/15`, `2635/7365/10000`.
- NWPU random: `22/23`, `7700/23800/31500`.
- NWPU confusable: `22/23`, `7700/23800/31500`.

## Known Runtime Requirements

Some official training scripts require extra dependencies such as `tensorboard`.
Several methods also require a valid DINO ViT-B/16 pretrained checkpoint path before
full training can start. Dataset loading does not require these checkpoints.
