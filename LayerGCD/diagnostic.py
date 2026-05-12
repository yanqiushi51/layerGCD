"""
Diagnostic script for LayerGCD.

Evaluates the raw KMeans clustering accuracy of different DINO layers
to verify if they contain complementary fine-grained information.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.cluster_and_log_utils import log_accs_from_preds
from network import build_multilayer_dino


@torch.no_grad()
def extract_all_features(model, dataloader, layers, device='cuda'):
    """Extract and normalize features from specified layers."""
    model.eval()
    
    features = {l: [] for l in layers}
    targets = []
    
    for batch in tqdm(dataloader, desc='Extracting features'):
        if len(batch) == 4:
            images, labels, _, _ = batch
        else:
            images, labels, _ = batch

        images = images[0].to(device) if isinstance(images, list) else images.to(device)
        
        layer_feats = model(images, return_all_layers=True)
        
        for l in layers:
            features[l].append(F.normalize(layer_feats[l], dim=-1).cpu().numpy())
            
        targets.append(labels.cpu().numpy())
        
    # Concatenate
    for l in layers:
        features[l] = np.concatenate(features[l], axis=0)
    
    targets = np.concatenate(targets, axis=0)
    return features, targets


def run_diagnostics():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cub')
    parser.add_argument('--split_type', type=str, default='random', choices=['random', 'confusable'])
    parser.add_argument('--class_split_seed', type=int, default=0)
    parser.add_argument('--extract_layers', nargs='+', type=int, default=[7, 9, 11])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    
    args = parser.parse_args()
    args.eval_funcs = ['v2']
    args.interpolation = 3
    args.crop_pct = 0.875
    args = get_class_splits(args)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # In DINO ViT-B, 11 is the final block
    extract_layers = sorted(list(set(args.extract_layers)))
    if 12 in extract_layers:
        extract_layers.remove(12)
        if 11 not in extract_layers:
            extract_layers.append(11)
            
    if 11 not in extract_layers:
        extract_layers.append(11)
        
    print(f"Testing DINO blocks: {extract_layers}")
    
    # 1. Model
    model = build_multilayer_dino(pretrained=True, extract_layers=extract_layers).to(device)
    
    # 2. Data
    # For diagnostic, we use simple validation transform
    _, test_transform = get_transform(args.transform, image_size=224, args=args)
    # Get standard datasets
    _, _, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, test_transform, test_transform, args
    )
    
    # Evaluate on the same unlabelled training split used by SimGCD:
    # leftover old-class examples plus all novel-class examples, without augmentation.
    dataloader = DataLoader(unlabelled_train_examples_test, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # 3. Extract and evaluate
    features, targets = extract_all_features(model, dataloader, extract_layers, device)
    
    total_classes = len(args.train_classes) + len(args.unlabeled_classes)
    
    print("\n" + "="*50)
    print(f"Diagnostic Results: {args.dataset_name}")
    print(f"KMeans with K={total_classes}")
    print("="*50)
    
    # Function to run KMeans and eval
    def eval_features(name, feats):
        kmeans = KMeans(n_clusters=total_classes, random_state=0, n_init=10)
        preds = kmeans.fit_predict(feats)
        
        # mask points to Old classes
        mask_old = targets < len(args.train_classes)
        
        # We use SimGCD's log_accs_from_preds which handles Hungarian matching
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=targets, y_pred=preds, mask=mask_old,
            T=0, eval_funcs=args.eval_funcs, save_name=name,
            args=args, print_output=False
        )
        print(f"{name:>15} | All: {all_acc:.4f} | Old: {old_acc:.4f} | New: {new_acc:.4f}")
        return all_acc, old_acc, new_acc
    
    print(f"{'Method':>15} | {'All':^6} | {'Old':^6} | {'New':^6}")
    print("-" * 50)
    
    best_single_all = 0
    
    # A. Independent layers
    for l in extract_layers:
        name = f"Layer {l}"
        all_acc, _, _ = eval_features(name, features[l])
        best_single_all = max(best_single_all, all_acc)
        
    # B. Concatenated layers
    print("-" * 50)
    concat_feats = np.concatenate([features[l] for l in extract_layers], axis=1)
    concat_feats = F.normalize(torch.from_numpy(concat_feats), dim=-1).numpy()
    all_acc, _, _ = eval_features("All Concat", concat_feats)
    
    # C. Analysis
    print("=" * 50)
    if all_acc > best_single_all:
        print(f"SUCCESS: Concatenation improves performance over best single layer by {(all_acc - best_single_all):.4f}")
        print("This validates that mid-level layers contain complementary fine-grained information!")
    else:
        print(f"WARNING: Concatenation DID NOT improve performance over best single layer.")
        print("This suggests the semantic abstraction hierarchy might be redundant for this dataset.")

if __name__ == '__main__':
    run_diagnostics()
