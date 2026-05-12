import os


_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_default_dataset_dir = os.path.join(_repo_root, 'data')
_dataset_dir = os.environ.get('DATASET_DIR', _default_dataset_dir)

# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/disk/datasets'
cifar_100_root = '/disk/datasets'
herbarium_dataroot = '/disk/datasets/herbarium_19'
imagenet_root = '/disk/datasets/ILSVRC12'

cub_root = '/disk/datasets/ood_zoo/ood_data/CUB'

aircraft_root = '/disk/datasets/ood_zoo/ood_data/aircraft/fgvc-aircraft-2013b'

car_root = '/disk/datasets/ood_zoo/ood_data/stanford_car'
scars_meta_path = "/disk/datasets/ood_zoo/ood_data/stanford_car/devkit/cars_{}.mat"
aid_root = os.path.join(_dataset_dir, 'AID')
nwpu_root = os.path.join(_dataset_dir, 'NWPU-RESISC45')

# OSR Split dir
osr_split_dir = '/disk/work/hjwang/osrd/data/open_set_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/disk/work/hjwang/pretrained_models/dino/dino_vitbase16_pretrain.pth' 
clip_pretrain_path = '/disk/work/hjwang/pretrained_models/clip/ViT-B-16.pt' 
feature_extract_dir = '/disk/work/hjwang/gcd/extracted_features_public_impl'     # Extract features to this directory
