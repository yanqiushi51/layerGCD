import os


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_default_dataset_dir = os.path.join(_repo_root, 'data')
_dataset_dir = os.environ.get('DATASET_DIR', _default_dataset_dir)

# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/data/dataset/cifar10'
cifar_100_root = '/data/dataset/cifar100'
cub_root = '/data/dataset/cub'
aircraft_root = '/data/dataset/fgvc-aircraft-2013b'
car_root = '/data/dataset/Stanford_Cars'
herbarium_dataroot = '/data/dataset/herbarium_19'
imagenet_root = '/data/dataset/imagenet'
aid_root = os.path.join(_dataset_dir, 'AID')
nwpu_root = os.path.join(_dataset_dir, 'NWPU-RESISC45')
# imagenet_1k = 'ImageNet/train/ILSVRC2012_img_train/data'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here
 
