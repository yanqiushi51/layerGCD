import os

# -----------------
# DATASET ROOTS
# -----------------
_dataset_dir = os.environ.get('DATASET_DIR', '/home/yqs/redar/data')

cifar_10_root = '/work/sagar/datasets/cifar10'
cifar_100_root = '/work/sagar/datasets/cifar100'
cub_root = '/work/sagar/datasets/CUB'
aircraft_root = '/work/khan/datasets/aircraft/fgvc-aircraft-2013b'
herbarium_dataroot = '/work/sagar/datasets/herbarium_19/'
imagenet_root = '/scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12'
aid_root = os.path.join(_dataset_dir, 'AID')
nwpu_root = os.path.join(_dataset_dir, 'NWPU-RESISC45')

# OSR Split dir
osr_split_dir = '/users/sagar/kai_collab/osr_novel_categories/data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/work/sagar/pretrained_models/dino/dino_vitbase16_pretrain.pth'
feature_extract_dir = '/work/sagar/osr_novel_categories/extracted_features_public_impl'     # Extract features to this directory
exp_root = '/work/sagar/osr_novel_categories/'          # All logs and checkpoints will be saved here
