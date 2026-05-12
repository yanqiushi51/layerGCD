"""
Main training script for LayerGCD.

Combines DINO multi-layer feature hierarchy with 
hierarchical clustering tree for Generalized Category Discovery.
"""

import argparse
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root

from model import PromptGuidedDINO, SupConLoss, info_nce_logits, DistillLoss, ContrastiveLearningViewGenerator
from hierarchy import HierarchicalClusterTree


def get_params_groups(model, base_lr=0.1, backbone_lr_mult=0.001):
    """Use a smaller LR for the unfrozen DINO block than for prompts/heads."""
    head_reg = []
    head_not_reg = []
    backbone_reg = []
    backbone_not_reg = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "dino_feature_extractor.backbone" in name or name.startswith("dino.")
        if name.endswith(".bias") or len(param.shape) == 1:
            (backbone_not_reg if is_backbone else head_not_reg).append(param)
        else:
            (backbone_reg if is_backbone else head_reg).append(param)
    
    groups = []
    if head_reg:
        groups.append({'params': head_reg, 'lr': base_lr})
    if head_not_reg:
        groups.append({'params': head_not_reg, 'lr': base_lr, 'weight_decay': 0.})

    backbone_lr = base_lr * backbone_lr_mult
    if backbone_reg:
        groups.append({'params': backbone_reg, 'lr': backbone_lr})
    if backbone_not_reg:
        groups.append({'params': backbone_not_reg, 'lr': backbone_lr, 'weight_decay': 0.})

    return groups


def get_num_clusters_per_level(extract_layers, total_classes, min_classes=8):
    """Mirror HierarchicalClusterTree's cluster schedule for model construction."""
    n_clusters_per_level = {}
    for i, layer_idx in enumerate(reversed(extract_layers)):
        n_clusters_per_level[layer_idx] = min(total_classes, max(total_classes // (2 ** i), min_classes))
    return n_clusters_per_level


def train(model, train_loader, eval_loader_unlabelled, extract_loader, args):
    device = next(model.parameters()).device
    params_groups = get_params_groups(
        model,
        base_lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
    )
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs, args.epochs, args.n_views,
        args.warmup_teacher_temp, args.teacher_temp
    )
    
    hierarchy_tree = None
    if not args.disable_hierarchy:
        hierarchy_tree = HierarchicalClusterTree(
            extract_layers=args.hierarchy_layers,
            n_labeled=args.num_labeled_classes,
            n_unlabeled=args.num_unlabeled_classes,
            min_classes=args.hierarchy_min_classes
        )
        args.logger.info("Building initial hierarchy tree (coarse targets)...")
        hierarchy_tree.build_hierarchy(model.dino_feature_extractor, extract_loader, device=device)
    else:
        args.logger.info("Hierarchy disabled: skipping coarse pseudo-targets and relation relaxation.")

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        if args.disable_hierarchy or args.fine_prompt_only or args.no_prompts or args.coarse_loss_weight <= 0:
            lambda_fine = 1.0
        elif args.curriculum_epochs > 0 and epoch < args.curriculum_epochs:
            lambda_fine = 0.0
        elif args.curriculum_ramp_epochs > 0:
            lambda_fine = min(1.0, (epoch - args.curriculum_epochs) / args.curriculum_ramp_epochs)
        else:
            lambda_fine = 1.0

        # Dynamic Hierarchy Update: re-cluster with evolved features
        if (
            hierarchy_tree is not None
            and args.hierarchy_rebuild_interval > 0
            and epoch > 0
            and epoch % args.hierarchy_rebuild_interval == 0
        ):
            args.logger.info(f"Epoch {epoch}: Re-building hierarchy tree with updated features...")
            hierarchy_tree.build_hierarchy(model.dino_feature_extractor, extract_loader, device=device)

        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels = class_labels.to(device, non_blocking=True)
            mask_lab = mask_lab.to(device, non_blocking=True).bool()
            
            # Combine augmented views into a single batch.
            images = torch.cat(images, dim=0).to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                # 1. Forward pass: Prompts
                coarse_logits, fine_logits, coarse_feat, fine_feat, coarse_proj, fine_proj = model(images)
                
                teacher_out_fine = fine_logits.detach()

                loss = fine_logits.new_zeros(())
                pstr = f'[Curriculum: {lambda_fine:.2f}] '
                
                # ==========================================
                # A: Coarse Level Losses (Layer K e.g. 7)
                # ==========================================
                if hierarchy_tree is not None and coarse_logits is not None and args.coarse_loss_weight > 0:
                    coarsest_layer = args.hierarchy_layers[0]
                    coarse_pseudo_labels = hierarchy_tree.get_pseudo_labels(
                        coarsest_layer, uq_idxs
                    ).repeat(args.n_views).to(class_labels.device)

                    loss_coarse = nn.CrossEntropyLoss()(coarse_logits / 0.1, coarse_pseudo_labels)

                    pstr += f'loss_c: {loss_coarse.item():.3f} '
                    loss += args.coarse_loss_weight * loss_coarse
                else:
                    pstr += 'loss_c: off '

                # ==========================================
                # B: Fine Level Losses (curriculum-gated)
                # ==========================================
                if lambda_fine > 0:
                    # 2. Unsupervised clustering (DistillLoss)
                    cluster_loss = cluster_criterion(fine_logits, teacher_out_fine, epoch)
                    
                    # ME-MAX (entropy maximization)
                    avg_probs = (fine_logits / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = math.log(float(len(avg_probs))) + torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
                    cluster_loss += args.memax_weight * me_max_loss
                    
                    # 3. InfoNCE with semantic-aware repulsion
                    confusion = None
                    if hierarchy_tree is not None and not args.disable_relation_relaxation:
                        confusion = hierarchy_tree.get_confusion_weights(
                            None,
                            uq_idxs,
                            device=fine_proj.device,
                            n_views=args.n_views,
                            mode=args.relation_relaxation_mode,
                        )
                    contrastive_logits, contrastive_labels = info_nce_logits(
                        features=fine_proj,
                        n_views=args.n_views,
                        device=fine_proj.device,
                        confusion_factor=confusion,
                        temperature=args.contrastive_temp,
                    )
                    contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    if mask_lab.any():
                        # 1. Supervised classification (labeled only)
                        sup_logits = torch.cat(
                            [f[mask_lab] for f in (fine_logits / 0.1).chunk(args.n_views)],
                            dim=0,
                        )
                        sup_labels = torch.cat(
                            [class_labels[mask_lab] for _ in range(args.n_views)],
                            dim=0,
                        )
                        cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                        # 4. Supervised contrastive (SupCon)
                        sp_chunked = torch.cat(
                            [f[mask_lab].unsqueeze(1) for f in fine_proj.chunk(args.n_views)],
                            dim=1,
                        )
                        sp_normed = F.normalize(sp_chunked, dim=-1)
                        sup_con_loss_global = SupConLoss()(sp_normed, labels=class_labels[mask_lab])
                    else:
                        cls_loss = fine_logits.new_zeros(())
                        sup_con_loss_global = fine_logits.new_zeros(())
                    
                    fine_loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    fine_loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss_global
                    
                    loss += lambda_fine * fine_loss
                    
                    pstr += f'cls_f: {cls_loss.item():.3f} clu_f: {cluster_loss.item():.3f} '
                    pstr += f'con_f: {contrastive_loss.item():.3f} sup_f: {sup_con_loss_global.item():.3f} '
                else:
                    pstr += 'fine: off '
                
                # Optimization step
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Evaluate on the unlabelled training examples only.
        args.logger.info('Testing on unlabelled training examples only...')
        all_acc, old_acc, new_acc = test(
            model,
            eval_loader_unlabelled,
            epoch=epoch,
            save_name='Train ACC Unlabelled Examples',
            args=args,
        )
        
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        exp_lr_scheduler.step()

        # Save Checkpoint
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'config': {
                'extract_layers': args.extract_layers,
                'num_coarse_classes': args.num_coarse_classes,
                'num_prompt_tokens': args.num_prompt_tokens,
                'hierarchy_min_classes': args.hierarchy_min_classes,
                'num_labeled_classes': args.num_labeled_classes,
                'num_unlabeled_classes': args.num_unlabeled_classes,
                'dataset_name': args.dataset_name,
                'split_type': args.split_type,
                'class_split_seed': args.class_split_seed,
                'hierarchy_layers': args.hierarchy_layers,
                'backbone_lr_mult': args.backbone_lr_mult,
                'coarse_loss_weight': args.coarse_loss_weight,
                'contrastive_temp': args.contrastive_temp,
                'curriculum_epochs': args.curriculum_epochs,
                'curriculum_ramp_epochs': args.curriculum_ramp_epochs,
                'hierarchy_rebuild_interval': args.hierarchy_rebuild_interval,
                'disable_hierarchy': args.disable_hierarchy,
                'single_layer_hierarchy': args.single_layer_hierarchy,
                'disable_bridge': args.disable_bridge,
                'fine_prompt_only': args.fine_prompt_only,
                'no_prompts': args.no_prompts,
                'disable_relation_relaxation': args.disable_relation_relaxation,
                'relation_relaxation_mode': args.relation_relaxation_mode,
            }
        }
        torch.save(save_dict, args.model_path)


def test(model, test_loader, epoch, save_name, args):
    model.eval()
    device = next(model.parameters()).device

    preds, targets = [], []
    mask = np.array([])
    
    for batch in tqdm(test_loader):
        if len(batch) == 4:
            images, label, _, _ = batch
        else:
            images, label, _ = batch
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            coarse_logits, logits, coarse_feat, fine_feat, coarse_proj, fine_proj = model(images)
            
            global_logits = logits
            preds.append(global_logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            
            label_np = label.cpu().numpy()
            mask = np.append(mask, label_np < args.num_labeled_classes)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LayerGCD')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--dataset_name', type=str, default='cub',
                        help='options: cifar10, cifar100, cub, scars, aircraft, herbarium_19, aid, nwpu')
    parser.add_argument('--split_type', type=str, default='random', choices=['random', 'confusable'],
                        help='Class split for AID/NWPU. Ignored by existing generic GCD datasets.')
    parser.add_argument('--class_split_seed', type=int, default=0,
                        help='Seed for random old/novel class split on AID/NWPU.')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    # Multi-layer arguments
    parser.add_argument('--extract_layers', nargs='+', type=int, default=[7, 9, 11, 12], 
                        help='DINO block indices to extract CLS token from')
    parser.add_argument('--grad_from_block', type=int, default=7,
                        help='Unused in prompt-guided mode; DINO is fully frozen')
    parser.add_argument('--hierarchy_update_interval', type=int, default=5,
                        help='Unused in prompt-guided mode; hierarchy is built once')
    parser.add_argument('--hierarchy_min_classes', type=int, default=8,
                        help='Minimum number of hierarchy clusters at the coarsest level')
    parser.add_argument('--num_prompt_tokens', type=int, default=4,
                        help='Number of learnable prompt tokens per branch')
    parser.add_argument('--disable_hierarchy', action='store_true', default=False,
                        help='Skip hierarchy pseudo-targets and hierarchy-aware relation relaxation.')
    parser.add_argument('--single_layer_hierarchy', action='store_true', default=False,
                        help='Use only the deepest DINO layer to build pseudo-targets.')
    parser.add_argument('--disable_bridge', action='store_true', default=False,
                        help='Keep coarse/fine prompts but block coarse-to-fine prompt transfer.')
    parser.add_argument('--fine_prompt_only', action='store_true', default=False,
                        help='Remove the coarse prompt branch and train only the fine prompt.')
    parser.add_argument('--no_prompts', action='store_true', default=False,
                        help='Remove all prompt tokens and train the classifier on DINO CLS only.')
    parser.add_argument('--disable_relation_relaxation', action='store_true', default=False,
                        help='Use standard InfoNCE without hierarchy-aware confusion weights.')
    parser.add_argument('--relation_relaxation_mode', type=str, default='multi', choices=['multi', 'coarse'],
                        help='Use all hierarchy levels or only the coarsest level for relation relaxation.')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--coarse_loss_weight', type=float, default=0.5)
    parser.add_argument('--contrastive_temp', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--curriculum_epochs', default=10, type=int)
    parser.add_argument('--curriculum_ramp_epochs', default=10, type=int)
    parser.add_argument('--hierarchy_rebuild_interval', default=0, type=int)
    parser.add_argument('--uniform_sampler', action='store_true', default=False)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['layergcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # MODELS
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768

    # Ensure extract layers are unique and sorted
    args.extract_layers = sorted(list(set(args.extract_layers)))
    
    # In DINO ViT-B, final block is index 11 (0 to 11). "Layer 12" often means block 11
    # Adjust indexing if 12 is passed
    if 12 in args.extract_layers:
        args.extract_layers.remove(12)
        if 11 not in args.extract_layers:
            args.extract_layers.append(11)

    args.logger.info(f"Extracting DINO layers: {args.extract_layers}")

    total_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.hierarchy_layers = [max(args.extract_layers)] if args.single_layer_hierarchy else list(args.extract_layers)
    args.num_clusters_per_level = get_num_clusters_per_level(
        args.hierarchy_layers,
        total_classes,
        min_classes=args.hierarchy_min_classes,
    )
    args.num_coarse_classes = args.num_clusters_per_level[args.hierarchy_layers[0]]
    args.logger.info(
        f"Split: {args.dataset_name}/{args.split_type} seed={args.class_split_seed} | "
        f"old={args.num_labeled_classes} novel={args.num_unlabeled_classes}"
    )
    args.logger.info(
        f"Hierarchy layers: {args.hierarchy_layers} | schedule: {args.num_clusters_per_level} | "
        f"coarse classes: {args.num_coarse_classes}"
    )
    
    model = PromptGuidedDINO(
        num_classes=total_classes,
        extract_layers=args.extract_layers,
        num_coarse_classes=args.num_coarse_classes,
        num_prompt_tokens=args.num_prompt_tokens,
        disable_bridge=args.disable_bridge,
        fine_prompt_only=args.fine_prompt_only,
        no_prompts=args.no_prompts,
    ).to(device)

    args.logger.info('Models built')

    # --------------------
    # DATA PIPELINE
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform_view = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform_view, test_transform, args
    )

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    # Balanced sampling between labelled old-class data and unlabelled GCD data.
    if args.uniform_sampler:
        sample_weights = [1.0 for _ in range(len(train_dataset))]
    else:
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                              shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)

    import copy
    extract_dataset = copy.deepcopy(train_dataset)
    extract_dataset.labelled_dataset.transform = test_transform
    extract_dataset.unlabelled_dataset.transform = test_transform
    extract_loader = DataLoader(extract_dataset, num_workers=args.num_workers,
                                batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # TRAIN
    # ----------------------
    # We only build the hierarchy purely as a prior targets cache ONCE, 
    # since DINO is frozen.
    args.logger.info("Initializing baseline DINO backbone and testing data flow...")
    train(model, train_loader, test_loader_unlabelled, extract_loader, args)
