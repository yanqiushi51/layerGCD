import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.get_datasets import get_class_splits, get_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['aid', 'nwpu'], required=True)
    parser.add_argument('--split_type', choices=['random', 'confusable'], default='random')
    parser.add_argument('--class_split_seed', type=int, default=0)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    args = parser.parse_args()

    args = get_class_splits(args)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name,
        train_transform=None,
        test_transform=None,
        args=args,
    )

    print(f"dataset={args.dataset_name} split_type={args.split_type} seed={args.class_split_seed}")
    print(f"old={len(args.train_classes)} novel={len(args.unlabeled_classes)}")
    print(f"train_labelled={len(datasets['train_labelled'])}")
    print(f"train_unlabelled={len(datasets['train_unlabelled'])}")
    print(f"train_merged={len(train_dataset)}")
    print(f"unlabelled_train_examples_test={len(unlabelled_train_examples_test)}")
    print(f"test={len(test_dataset)}")


if __name__ == '__main__':
    main()
