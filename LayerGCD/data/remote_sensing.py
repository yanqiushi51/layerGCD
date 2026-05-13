import os
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension

from config import aid_root, nwpu_root
from data.data_utils import subsample_instances


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

AID_CLASSES = [
    "airport",
    "bare land",
    "baseball field",
    "beach",
    "bridge",
    "center",
    "church",
    "commercial",
    "dense residential",
    "desert",
    "farmland",
    "forest",
    "industrial",
    "meadow",
    "medium residential",
    "mountain",
    "park",
    "parking",
    "playground",
    "pond",
    "port",
    "railway station",
    "resort",
    "river",
    "school",
    "sparse residential",
    "square",
    "stadium",
    "storage tanks",
    "viaduct",
]

NWPU_CLASSES = [
    "airplane",
    "airport",
    "baseball diamond",
    "basketball court",
    "beach",
    "bridge",
    "chaparral",
    "church",
    "circular farmland",
    "cloud",
    "commercial area",
    "dense residential",
    "desert",
    "forest",
    "freeway",
    "golf course",
    "ground track field",
    "harbor",
    "industrial area",
    "intersection",
    "island",
    "lake",
    "meadow",
    "medium residential",
    "mobile home park",
    "mountain",
    "overpass",
    "palace",
    "parking lot",
    "railway",
    "railway station",
    "rectangular farmland",
    "river",
    "roundabout",
    "runway",
    "sea ice",
    "ship",
    "snowberg",
    "sparse residential",
    "stadium",
    "storage tank",
    "tennis court",
    "terrace",
    "thermal power station",
    "wetland",
]

AID_CONFUSABLE_OLD = [
    "airport",
    "baseball field",
    "beach",
    "bridge",
    "commercial",
    "dense residential",
    "farmland",
    "forest",
    "industrial",
    "port",
    "river",
    "school",
    "stadium",
    "storage tanks",
    "viaduct",
]

AID_CONFUSABLE_NOVEL = [
    "bare land",
    "center",
    "church",
    "desert",
    "meadow",
    "medium residential",
    "mountain",
    "park",
    "parking",
    "playground",
    "pond",
    "railway station",
    "resort",
    "sparse residential",
    "square",
]

NWPU_CONFUSABLE_OLD = [
    "airport",
    "baseball diamond",
    "basketball court",
    "beach",
    "bridge",
    "chaparral",
    "circular farmland",
    "cloud",
    "commercial area",
    "dense residential",
    "desert",
    "forest",
    "freeway",
    "golf course",
    "harbor",
    "lake",
    "mountain",
    "overpass",
    "railway",
    "stadium",
    "storage tank",
    "tennis court",
]

NWPU_CONFUSABLE_NOVEL = [
    "airplane",
    "church",
    "ground track field",
    "industrial area",
    "intersection",
    "island",
    "meadow",
    "medium residential",
    "mobile home park",
    "palace",
    "parking lot",
    "railway station",
    "rectangular farmland",
    "river",
    "roundabout",
    "runway",
    "sea ice",
    "ship",
    "snowberg",
    "sparse residential",
    "terrace",
    "thermal power station",
    "wetland",
]

DATASET_META = {
    "aid": {
        "root": aid_root,
        "classes": AID_CLASSES,
        "num_old": 15,
        "confusable_old": AID_CONFUSABLE_OLD,
        "confusable_novel": AID_CONFUSABLE_NOVEL,
    },
    "nwpu": {
        "root": nwpu_root,
        "classes": NWPU_CLASSES,
        "num_old": 22,
        "confusable_old": NWPU_CONFUSABLE_OLD,
        "confusable_novel": NWPU_CONFUSABLE_NOVEL,
    },
}


def _normalize_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _class_name_to_idx(classes):
    return {name: idx for idx, name in enumerate(classes)}


def _resolve_root(root):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(root)))


def _find_class_dirs(root, class_names):
    root = _resolve_root(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Remote sensing dataset root does not exist: {root}")

    expected = {_normalize_name(name): name for name in class_names}

    def scan(candidate_root):
        found = {}
        for entry in os.scandir(candidate_root):
            if not entry.is_dir():
                continue
            key = _normalize_name(entry.name)
            if key in expected:
                found[expected[key]] = entry.path
        return found

    found = scan(root)
    if len(found) == len(class_names):
        return found

    for entry in os.scandir(root):
        if entry.is_dir():
            nested = scan(entry.path)
            if len(nested) > len(found):
                found = nested
            if len(found) == len(class_names):
                return found

    missing = sorted(set(class_names) - set(found))
    raise RuntimeError(
        f"Could not match all class folders under {root}. Missing: {missing}. "
        "Folder names may use spaces, underscores, hyphens, or CamelCase."
    )


class RemoteSensingSceneDataset(Dataset):
    def __init__(self, root, class_names, include_classes=None, transform=None,
                 target_transform=None, loader=default_loader):
        self.root = _resolve_root(root)
        self.class_names = list(class_names)
        self.include_classes = set(self.class_names if include_classes is None else include_classes)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        class_dirs = _find_class_dirs(self.root, self.class_names)
        name_to_idx = _class_name_to_idx(self.class_names)
        samples = []
        for class_name in self.class_names:
            if class_name not in self.include_classes:
                continue
            class_dir = class_dirs[class_name]
            target = name_to_idx[class_name]
            for dirpath, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    path = os.path.join(dirpath, filename)
                    if has_file_allowed_extension(path, IMG_EXTENSIONS):
                        samples.append((path, target))

        if not samples:
            raise RuntimeError(f"No images found for classes {sorted(self.include_classes)} in {self.root}")

        self.samples = sorted(samples)
        self.uq_idxs = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, self.uq_idxs[idx]


def _subsample_dataset(dataset, idxs):
    dataset.samples = [dataset.samples[i] for i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def _stratified_train_test_indices(dataset, train_ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    by_class = {}
    for idx, (_, target) in enumerate(dataset.samples):
        by_class.setdefault(target, []).append(idx)

    train_indices = []
    test_indices = []
    for target in sorted(by_class):
        idxs = np.array(by_class[target])
        idxs = idxs[rng.permutation(len(idxs))]
        n_train = int(round(train_ratio * len(idxs)))
        n_train = min(max(n_train, 1), len(idxs) - 1)
        train_indices.extend(idxs[:n_train].tolist())
        test_indices.extend(idxs[n_train:].tolist())

    return np.array(sorted(train_indices)), np.array(sorted(test_indices))


def get_remote_sensing_class_splits(dataset_name, split_type="random", seed=0):
    meta = DATASET_META[dataset_name]
    classes = meta["classes"]

    if split_type == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(classes))
        old = sorted(perm[:meta["num_old"]].tolist())
        novel = sorted(perm[meta["num_old"]:].tolist())
    elif split_type == "confusable":
        name_to_idx = _class_name_to_idx(classes)
        old = [name_to_idx[name] for name in meta["confusable_old"]]
        novel = [name_to_idx[name] for name in meta["confusable_novel"]]
    else:
        raise ValueError(f"Unknown split_type '{split_type}'. Use 'random' or 'confusable'.")

    if len(set(old) & set(novel)) > 0:
        raise RuntimeError(f"{dataset_name} {split_type} split has overlapping old/novel classes")
    if sorted(old + novel) != list(range(len(classes))):
        raise RuntimeError(f"{dataset_name} {split_type} split does not cover all classes")

    return old, novel


def get_remote_sensing_datasets(dataset_name, train_transform, test_transform,
                                train_classes, prop_train_labels=0.5,
                                split_train_val=False, seed=0,
                                train_ratio=0.7, image_split_seed=0):
    if split_train_val:
        raise NotImplementedError("Remote sensing datasets use the GCD train/unlabelled protocol only.")

    meta = DATASET_META[dataset_name]
    root = meta["root"]
    class_names = meta["classes"]

    whole_dataset = RemoteSensingSceneDataset(
        root=root,
        class_names=class_names,
        transform=train_transform,
    )
    train_indices, test_indices = _stratified_train_test_indices(
        whole_dataset,
        train_ratio=train_ratio,
        seed=image_split_seed,
    )
    train_pool = _subsample_dataset(deepcopy(whole_dataset), train_indices)

    old_class_set = set(train_classes)
    old_indices = [
        idx for idx, (_, target) in enumerate(train_pool.samples)
        if target in old_class_set
    ]
    labelled_dataset = _subsample_dataset(deepcopy(train_pool), np.array(old_indices))
    subsample_indices = subsample_instances(
        labelled_dataset,
        prop_indices_to_subsample=prop_train_labels,
    )
    labelled_dataset = _subsample_dataset(labelled_dataset, subsample_indices)

    labelled_global_paths = {path for path, _ in labelled_dataset.samples}
    unlabelled_indices = [
        idx for idx, (path, _) in enumerate(train_pool.samples)
        if path not in labelled_global_paths
    ]
    unlabelled_dataset = _subsample_dataset(deepcopy(train_pool), np.array(unlabelled_indices))
    test_dataset = _subsample_dataset(deepcopy(whole_dataset), test_indices)
    test_dataset.transform = test_transform

    return {
        "train_labelled": labelled_dataset,
        "train_unlabelled": unlabelled_dataset,
        "val": None,
        "test": test_dataset,
    }
