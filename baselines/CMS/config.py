# -----------------
# DATASET ROOTS
# -----------------
import os

DATASET_ROOT = os.environ.get('DATASET_DIR', '/home/yqs/redar/data')
if not DATASET_ROOT.endswith(os.sep):
    DATASET_ROOT += os.sep
cifar_10_root = DATASET_ROOT+'cifar10'
cifar_100_root = DATASET_ROOT+'cifar100'
cub_root = DATASET_ROOT
aircraft_root = DATASET_ROOT+'fgvc-aircraft-2013b'
herbarium_dataroot = DATASET_ROOT+'herbarium_19'
imagenet_root = DATASET_ROOT+'imagenet'
aid_root = os.path.join(DATASET_ROOT, 'AID')
nwpu_root = os.path.join(DATASET_ROOT, 'NWPU-RESISC45')

# OSR Split dir
osr_split_dir = './data/ssb_splits'

dino_pretrain_path = './models/dino_vitbase16_pretrain.pth'
exp_root = './log'          # All logs and checkpoints will be saved here
