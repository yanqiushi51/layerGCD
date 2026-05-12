import os


_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_dataset_dir = os.path.join(_repo_root, 'data')
_dataset_dir = os.environ.get('DATASET_DIR', _default_dataset_dir)

# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '${DATASET_DIR}/cifar10'
cifar_100_root = '${DATASET_DIR}/cifar100'
cub_root = '/root/Cold-Discovery/CUB_200_2011'
aircraft_root = '${DATASET_DIR}/fgvc-aircraft-2013b'
car_root = '${DATASET_DIR}/cars'
herbarium_dataroot = '${DATASET_DIR}/herbarium_19'
imagenet_root = '${DATASET_DIR}/ImageNet'
aid_root = os.path.join(_dataset_dir, 'AID')
nwpu_root = os.path.join(_dataset_dir, 'NWPU-RESISC45')

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = os.environ.get(
    'EXP_ROOT',
    os.path.join(_repo_root, 'LayerGCD', 'dev_outputs'),
)  # All logs and checkpoints will be saved here
