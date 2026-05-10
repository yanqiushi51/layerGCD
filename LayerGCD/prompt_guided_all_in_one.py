"""
All-in-one reference for the current Prompt-Guided LayerGCD pipeline.

This file consolidates the active prompt-guided path into one place so it is
easy to hand to a browser-based AI reviewer. It keeps the current codebase's
behavior:

- frozen DINO ViT-B/16 backbone
- two learnable prompts: P_coarse and P_fine
- coarse prior injection through bridge_mlp
- hierarchy-built coarse pseudo-labels
- curriculum learning: coarse first, fine later
- fine branch evaluated as the final classifier

Dataset helpers and logging utilities are still imported from the local
codebase to avoid duplicating all dataset definitions.
"""

import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import exp_root
from data.augmentations import get_transform
from data.get_datasets import get_class_splits, get_datasets
from util.cluster_and_log_utils import log_accs_from_preds
from util.general_utils import AverageMeter, init_experiment


class MultiLayerDINO(nn.Module):
    """Frozen DINO wrapper used only for hierarchy feature extraction."""

    def __init__(self, backbone, extract_layers=(7, 9, 11)):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = sorted(extract_layers)
        self.depth = len(backbone.blocks)
        if (self.depth - 1) not in self.extract_layers:
            self.extract_layers.append(self.depth - 1)
        self.feat_dim = backbone.embed_dim

    def forward(self, x, return_all_layers=False):
        x = self.backbone.prepare_tokens(x)
        layer_features = {}

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if return_all_layers and i in self.extract_layers:
                layer_features[i] = self.backbone.norm(x)[:, 0]

        x = self.backbone.norm(x)
        final_cls = x[:, 0]

        if return_all_layers:
            layer_features[self.depth - 1] = final_cls
            return layer_features
        return final_cls


def build_multilayer_dino(pretrained=True, extract_layers=(7, 9, 11), grad_from_block=7):
    """Build the local DINO wrapper and apply the same freezing logic as the repo."""
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=pretrained)

    for param in backbone.parameters():
        param.requires_grad = False

    for name, param in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= grad_from_block:
                param.requires_grad = True

    return MultiLayerDINO(backbone, extract_layers=list(extract_layers))


class DINOHead(nn.Module):
    """Projection + classifier head used by both prompts."""

    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        proj = self.mlp(x)
        logits = self.last_layer(F.normalize(x, dim=-1, p=2))
        return proj, logits


class PromptGuidedDINO(nn.Module):
    """
    Current prompt-guided model used in LayerGCD.

    Important behavior:
    - DINO is fully frozen
    - P_coarse is read at the coarse checkpoint
    - bridge_mlp(P_coarse) is injected into P_fine
    - P_fine is used for the final fine-grained prediction
    """

    def __init__(
        self,
        num_classes,
        extract_layers=(7, 11),
        num_coarse_classes=None,
        num_prompt_tokens=4,
    ):
        super().__init__()
        extract_layers = sorted(set(extract_layers))
        if 12 in extract_layers:
            extract_layers.remove(12)
            extract_layers.append(11)
        extract_layers = sorted(set(extract_layers))
        self.dino_feature_extractor = build_multilayer_dino(
            pretrained=True,
            extract_layers=extract_layers,
            grad_from_block=11,  # Unfreeze only Block 11
        )
        self.dino = self.dino_feature_extractor.backbone

        self.num_prompt_tokens = num_prompt_tokens
        self.P_coarse = nn.Parameter(torch.randn(1, num_prompt_tokens, 768))
        self.P_fine = nn.Parameter(torch.randn(1, num_prompt_tokens, 768))
        nn.init.normal_(self.P_coarse, std=0.02)
        nn.init.normal_(self.P_fine, std=0.02)

        self.coarse_layer_idx = extract_layers[0]
        self.fine_layer_idx = extract_layers[-1]

        if num_coarse_classes is None:
            num_coarse_classes = num_classes

        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_classes

        self.coarse_head = DINOHead(in_dim=768, out_dim=num_coarse_classes, nlayers=3)
        # Classification: CLS (768) + P_fine (768) = 1536
        self.fine_head = DINOHead(in_dim=768 * 2, out_dim=num_classes, nlayers=3)

        # V4: Separate contrastive head — CLS only (768 → 256), clean view-invariant projection
        self.contrastive_head = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
        )

        # V5: Prompt-aware contrastive head — P_fine only (768 → 256)
        self.prompt_contrastive_head = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
        )

        self.bridge_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )
        # Zero-init last layer so bridge starts as identity (no noise injection)
        nn.init.constant_(self.bridge_mlp[-1].weight, 0)
        nn.init.constant_(self.bridge_mlp[-1].bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        num_prompt_tokens = self.num_prompt_tokens
        x = self.dino.prepare_tokens(x)

        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        p_coarse = self.P_coarse.expand(batch_size, -1, -1)
        p_fine = self.P_fine.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, p_coarse, p_fine, patch_tokens], dim=1)

        coarse_feat = None
        coarse_logits = None
        coarse_proj = None

        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)

            if i == self.coarse_layer_idx:
                coarse_raw = x[:, 1 : 1 + num_prompt_tokens, :].mean(dim=1)
                coarse_feat = self.dino.norm(coarse_raw)
                coarse_proj, coarse_logits = self.coarse_head(coarse_feat)

                prior_msg = self.bridge_mlp(coarse_feat)
                fine_start = 1 + num_prompt_tokens
                fine_end = 1 + 2 * num_prompt_tokens
                x_p_fine_new = x[:, fine_start:fine_end, :] + prior_msg.unsqueeze(1)
                x = torch.cat([x[:, :fine_start, :], x_p_fine_new, x[:, fine_end:, :]], dim=1)

        x = self.dino.norm(x)
        fine_start = 1 + num_prompt_tokens
        fine_end = 1 + 2 * num_prompt_tokens

        cls_feat = x[:, 0, :]  # [B, 768] — DINO's view-invariant global feature
        fine_prompt_feat = x[:, fine_start:fine_end, :].mean(dim=1)  # [B, 768]

        # V4 dual pathway:
        # Classification path: CLS + P_fine concat → logits
        fine_feat = torch.cat([cls_feat, fine_prompt_feat], dim=-1)  # [B, 1536]
        _, fine_logits = self.fine_head(fine_feat)

        # Contrastive path: CLS only → clean 256-dim projection
        fine_proj = self.contrastive_head(cls_feat)

        # V5: Prompt contrastive path: P_fine only → 256-dim projection
        prompt_proj = self.prompt_contrastive_head(fine_prompt_feat)

        return coarse_logits, fine_logits, coarse_feat, fine_feat, coarse_proj, fine_proj, prompt_proj


class HierarchicalClusterTree:
    """Hierarchy builder used to produce coarse pseudo-labels and confusion weights."""

    def __init__(self, extract_layers, n_labeled, n_unlabeled, min_classes=8):
        self.extract_layers = extract_layers
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.min_classes = min_classes

        total_classes = n_labeled + n_unlabeled
        self.n_clusters_per_level = {}
        for i, layer_idx in enumerate(reversed(extract_layers)):
            self.n_clusters_per_level[layer_idx] = max(total_classes // (2 ** i), min_classes)

        self.prototypes = {}
        self.pseudo_labels = {}
        self.cluster_radii = {}
        self.index_to_position = {}

    @torch.no_grad()
    def build_hierarchy(self, model, dataloader, device="cuda"):
        model.eval()
        all_features = {layer: [] for layer in self.extract_layers}
        all_indices = []

        for batch in tqdm(dataloader, desc="Extracting features for hierarchy"):
            if len(batch) == 4:
                images, _, uq_idxs, _ = batch
            else:
                images, _, uq_idxs = batch

            images = images[0].to(device) if isinstance(images, list) else images.to(device)
            layer_feats = model(images, return_all_layers=True)

            for layer_idx, feat in layer_feats.items():
                all_features[layer_idx].append(F.normalize(feat, dim=-1).cpu())
            all_indices.append(uq_idxs)

        for layer_idx in self.extract_layers:
            all_features[layer_idx] = torch.cat(all_features[layer_idx], dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        sort_order = torch.argsort(all_indices)
        all_indices = all_indices[sort_order]
        for layer_idx in self.extract_layers:
            all_features[layer_idx] = all_features[layer_idx][sort_order]

        self.index_to_position = {
            int(sample_idx): pos for pos, sample_idx in enumerate(all_indices.tolist())
        }

        deepest_layer = max(self.extract_layers)
        deepest_feats = all_features[deepest_layer].numpy()
        deepest_k = self.n_clusters_per_level[deepest_layer]
        kmeans = KMeans(n_clusters=deepest_k, random_state=0, n_init=10)
        preds = kmeans.fit_predict(deepest_feats)
        self.prototypes[deepest_layer] = torch.from_numpy(kmeans.cluster_centers_).float()
        self.pseudo_labels[deepest_layer] = torch.from_numpy(preds).long()
        self._compute_radii(deepest_layer, all_features[deepest_layer])

        prev_layer = deepest_layer
        for layer_idx in reversed(self.extract_layers[:-1]):
            feats_np = all_features[layer_idx].numpy()
            n_clusters = self.n_clusters_per_level[layer_idx]

            prev_protos_np = self.prototypes[prev_layer].numpy()
            proto_kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            proto_mapping = proto_kmeans.fit_predict(prev_protos_np)

            prev_pseudo = self.pseudo_labels[prev_layer].numpy()
            coarse_init_labels = np.array([proto_mapping[p] for p in prev_pseudo])

            init_protos = np.zeros((n_clusters, feats_np.shape[1]), dtype=feats_np.dtype)
            for c in range(n_clusters):
                mask_c = coarse_init_labels == c
                if mask_c.sum() > 0:
                    init_protos[c] = feats_np[mask_c].mean(axis=0)

            norms = np.linalg.norm(init_protos, axis=1, keepdims=True)
            norms[norms == 0] = 1
            init_protos = init_protos / norms

            layer_kmeans = KMeans(
                n_clusters=n_clusters,
                init=init_protos,
                n_init=1,
                random_state=0,
            )
            refined_preds = layer_kmeans.fit_predict(feats_np)

            self.prototypes[layer_idx] = torch.from_numpy(layer_kmeans.cluster_centers_).float()
            self.pseudo_labels[layer_idx] = torch.from_numpy(refined_preds).long()
            self._compute_radii(layer_idx, all_features[layer_idx])
            prev_layer = layer_idx

        model.train()

    def _compute_radii(self, layer_idx, features):
        pseudo = self.pseudo_labels[layer_idx]
        protos = self.prototypes[layer_idx]
        radii = torch.zeros(protos.shape[0])
        for c in range(protos.shape[0]):
            mask = pseudo == c
            if mask.sum() > 0:
                dists = torch.cdist(features[mask], protos[c : c + 1]).squeeze(1)
                radii[c] = dists.mean()
        self.cluster_radii[layer_idx] = radii

    def _resolve_positions(self, indices):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        return torch.tensor(
            [self.index_to_position[int(sample_idx)] for sample_idx in indices],
            dtype=torch.long,
        )

    def get_pseudo_labels(self, layer_idx, indices=None):
        labels = self.pseudo_labels[layer_idx]
        if indices is None:
            return labels
        return labels[self._resolve_positions(indices)]

    def get_confusion_weights(self, _features_dict, sample_indices, device="cuda", n_views=2):
        batch_size = len(sample_indices)
        confusion = torch.zeros(batch_size, batch_size, device=device)
        positions = self._resolve_positions(sample_indices)
        num_levels = len(self.extract_layers)

        for i, layer_idx in enumerate(self.extract_layers):
            pseudo = self.pseudo_labels[layer_idx][positions].to(device)
            same_cluster = (pseudo.unsqueeze(0) == pseudo.unsqueeze(1)).float()
            weight = 1.0 / (2 ** (num_levels - i - 1))
            confusion += weight * same_cluster

        confusion = confusion / confusion.max()
        return confusion.repeat(n_views, n_views)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


def info_nce_logits(features, n_views=2, temperature=1.0, device="cuda", confusion_factor=None):
    if features.size(0) % n_views != 0:
        raise ValueError("features.size(0) must be divisible by n_views")

    batch_size = int(features.size(0) // n_views)
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    if confusion_factor is not None:
        confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)
        similarity_matrix = similarity_matrix + 0.5 * confusion_factor

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    positive_logits = torch.logsumexp(positives / temperature, dim=1, keepdim=True)
    negative_logits = negatives / temperature

    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, labels


class DistillLoss(nn.Module):
    def __init__(
        self,
        warmup_teacher_temp_epochs,
        nepochs,
        ncrops=2,
        warmup_teacher_temp=0.07,
        teacher_temp=0.04,
        student_temp=0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1).detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        return total_loss / n_loss_terms


class ContrastiveLearningViewGenerator:
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for _ in range(self.n_views)]
        return [self.base_transform[i](x) for i in range(self.n_views)]


def get_params_groups(model, base_lr=0.1, backbone_lr_mult=0.001):
    """Use a much smaller LR for the unfrozen DINO block than for prompts/heads."""
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
        groups.append({"params": head_reg, "lr": base_lr})
    if head_not_reg:
        groups.append({"params": head_not_reg, "lr": base_lr, "weight_decay": 0.0})

    backbone_lr = base_lr * backbone_lr_mult
    if backbone_reg:
        groups.append({"params": backbone_reg, "lr": backbone_lr})
    if backbone_not_reg:
        groups.append({"params": backbone_not_reg, "lr": backbone_lr, "weight_decay": 0.0})

    return groups


def get_num_clusters_per_level(extract_layers, total_classes, min_classes=8):
    n_clusters_per_level = {}
    for i, layer_idx in enumerate(reversed(extract_layers)):
        n_clusters_per_level[layer_idx] = max(total_classes // (2 ** i), min_classes)
    return n_clusters_per_level


def train(model, train_loader, eval_loader_unlabelled, extract_loader, args):
    device = next(model.parameters()).device
    params_groups = get_params_groups(
        model,
        base_lr=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
    )
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    hierarchy_tree = HierarchicalClusterTree(
        extract_layers=args.extract_layers,
        n_labeled=args.num_labeled_classes,
        n_unlabeled=args.num_unlabeled_classes,
        min_classes=args.hierarchy_min_classes,
    )

    args.logger.info("Building initial hierarchy tree (coarse targets)...")
    hierarchy_tree.build_hierarchy(model.dino_feature_extractor, extract_loader, device=device)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        if args.curriculum_epochs > 0 and epoch < args.curriculum_epochs:
            lambda_fine = 0.0
        elif args.curriculum_ramp_epochs > 0:
            lambda_fine = min(1.0, (epoch - args.curriculum_epochs) / args.curriculum_ramp_epochs)
        else:
            lambda_fine = 1.0

        # Dynamic Hierarchy Update: re-cluster with evolved features
        if (
            args.hierarchy_rebuild_interval > 0
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
            images = torch.cat(images, dim=0).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                coarse_logits, fine_logits, _, _, _, fine_proj, prompt_proj = model(images)
                teacher_out_fine = fine_logits.detach()

                loss = 0
                pstr = f"[Curriculum: {lambda_fine:.2f}] "

                # --- Coarse loss ---
                coarsest_layer = args.extract_layers[0]
                coarse_pseudo_labels = (
                    hierarchy_tree.get_pseudo_labels(coarsest_layer, uq_idxs)
                    .repeat(args.n_views)
                    .to(class_labels.device)
                )
                loss_coarse = nn.CrossEntropyLoss()(coarse_logits / 0.1, coarse_pseudo_labels)
                pstr += f"loss_c: {loss_coarse.item():.3f} "
                loss += args.coarse_loss_weight * loss_coarse

                if lambda_fine > 0:
                    cluster_loss = cluster_criterion(fine_logits, teacher_out_fine, epoch)
                    avg_probs = (fine_logits / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = math.log(float(len(avg_probs))) + torch.sum(
                        avg_probs * torch.log(avg_probs + 1e-8)
                    )
                    cluster_loss += args.memax_weight * me_max_loss

                    confusion = hierarchy_tree.get_confusion_weights(
                        None,
                        uq_idxs,
                        device=fine_proj.device,
                        n_views=args.n_views,
                    )
                    contrastive_logits, contrastive_labels = info_nce_logits(
                        features=fine_proj,
                        n_views=args.n_views,
                        device=fine_proj.device,
                        confusion_factor=confusion,
                        temperature=args.contrastive_temp,
                    )
                    contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    prompt_con_logits, prompt_con_labels = info_nce_logits(
                        features=prompt_proj,
                        n_views=args.n_views,
                        device=prompt_proj.device,
                        temperature=args.contrastive_temp,
                    )
                    prompt_con_loss = nn.CrossEntropyLoss()(prompt_con_logits, prompt_con_labels)

                    if mask_lab.any():
                        sup_logits = torch.cat(
                            [f[mask_lab] for f in (fine_logits / 0.1).chunk(args.n_views)],
                            dim=0,
                        )
                        sup_labels = torch.cat(
                            [class_labels[mask_lab] for _ in range(args.n_views)],
                            dim=0,
                        )
                        cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                        sp_chunked = torch.cat(
                            [f[mask_lab].unsqueeze(1) for f in fine_proj.chunk(args.n_views)],
                            dim=1,
                        )
                        sp_normed = F.normalize(sp_chunked, dim=-1)
                        sup_con_loss = SupConLoss()(sp_normed, labels=class_labels[mask_lab])
                    else:
                        cls_loss = fine_logits.new_zeros(())
                        sup_con_loss = fine_logits.new_zeros(())

                    fine_loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                    fine_loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                    fine_loss += args.prompt_con_weight * prompt_con_loss

                    pstr += f"cls_f: {cls_loss.item():.3f} clu_f: {cluster_loss.item():.3f} "
                    pstr += f"con_f: {contrastive_loss.item():.3f} sup_f: {sup_con_loss.item():.3f} "
                    pstr += f"p_con: {prompt_con_loss.item():.3f} "

                    loss += lambda_fine * fine_loss
                else:
                    pstr += "fine: off "

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
                args.logger.info(
                    "Epoch: [{}][{}/{}]\t loss {:.5f}\t {}".format(
                        epoch, batch_idx, len(train_loader), loss.item(), pstr
                    )
                )

        args.logger.info("Train Epoch: {} Avg Loss: {:.4f} ".format(epoch, loss_record.avg))
        args.logger.info("Testing on unlabelled training examples only...")
        all_acc, old_acc, new_acc = test(
            model,
            eval_loader_unlabelled,
            epoch=epoch,
            save_name="Train ACC Unlabelled Examples",
            args=args,
        )
        args.logger.info(
            "Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(
                all_acc, old_acc, new_acc
            )
        )

        exp_lr_scheduler.step()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": {
                    "extract_layers": args.extract_layers,
                    "num_coarse_classes": args.num_coarse_classes,
                    "num_prompt_tokens": args.num_prompt_tokens,
                    "hierarchy_min_classes": args.hierarchy_min_classes,
                    "num_labeled_classes": args.num_labeled_classes,
                    "num_unlabeled_classes": args.num_unlabeled_classes,
                    "backbone_lr_mult": args.backbone_lr_mult,
                    "coarse_loss_weight": args.coarse_loss_weight,
                    "prompt_con_weight": args.prompt_con_weight,
                    "contrastive_temp": args.contrastive_temp,
                    "curriculum_epochs": args.curriculum_epochs,
                    "curriculum_ramp_epochs": args.curriculum_ramp_epochs,
                    "hierarchy_rebuild_interval": args.hierarchy_rebuild_interval,
                    "uniform_sampler": args.uniform_sampler,
                },
            },
            args.model_path,
        )


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
            _, fine_logits, *_ = model(images)
            preds.append(fine_logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            label_np = label.cpu().numpy()
            mask = np.append(mask, label_np < args.num_labeled_classes)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return log_accs_from_preds(
        y_true=targets,
        y_pred=preds,
        mask=mask,
        T=epoch,
        eval_funcs=args.eval_funcs,
        save_name=save_name,
        args=args,
    )


def main():
    parser = argparse.ArgumentParser(description="Prompt-Guided LayerGCD (all-in-one)")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_funcs", nargs="+", default=["v2"])
    parser.add_argument("--dataset_name", type=str, default="cub")
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", action="store_true", default=True)
    parser.add_argument("--extract_layers", nargs="+", type=int, default=[7, 9, 11, 12])
    parser.add_argument("--hierarchy_min_classes", default=8, type=int)
    parser.add_argument("--num_prompt_tokens", default=4, type=int)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--sup_weight", type=float, default=0.35)
    parser.add_argument("--coarse_loss_weight", type=float, default=0.5)
    parser.add_argument("--prompt_con_weight", type=float, default=0.3)
    parser.add_argument("--contrastive_temp", type=float, default=0.5)
    parser.add_argument("--n_views", default=2, type=int)
    parser.add_argument("--curriculum_epochs", default=10, type=int)
    parser.add_argument("--curriculum_ramp_epochs", default=10, type=int)
    parser.add_argument("--hierarchy_rebuild_interval", default=0, type=int)
    parser.add_argument("--uniform_sampler", action="store_true", default=False)
    parser.add_argument("--memax_weight", type=float, default=2.0)
    parser.add_argument("--warmup_teacher_temp", default=0.07, type=float)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=30, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--exp_name", default=None, type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=["layergcd_all_in_one"])
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")

    torch.backends.cudnn.benchmark = True
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768

    args.extract_layers = sorted(list(set(args.extract_layers)))
    if 12 in args.extract_layers:
        args.extract_layers.remove(12)
        if 11 not in args.extract_layers:
            args.extract_layers.append(11)

    total_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.num_clusters_per_level = get_num_clusters_per_level(
        args.extract_layers,
        total_classes,
        min_classes=args.hierarchy_min_classes,
    )
    args.num_coarse_classes = args.num_clusters_per_level[args.extract_layers[0]]

    args.logger.info(
        f"Extracting DINO layers: {args.extract_layers} | "
        f"Hierarchy cluster schedule: {args.num_clusters_per_level} | "
        f"coarse classes: {args.num_coarse_classes}"
    )
    model = PromptGuidedDINO(
        num_classes=total_classes,
        extract_layers=args.extract_layers,
        num_coarse_classes=args.num_coarse_classes,
        num_prompt_tokens=args.num_prompt_tokens,
    ).to(device)
    args.logger.info("Model built")

    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args
    )
    train_transform_view = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views
    )

    train_dataset, _, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, train_transform_view, test_transform, args
    )

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    if args.uniform_sampler:
        sample_weights = [1.0 for _ in range(len(train_dataset))]
    else:
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=False,
    )

    extract_dataset = copy.deepcopy(train_dataset)
    extract_dataset.labelled_dataset.transform = test_transform
    extract_dataset.unlabelled_dataset.transform = test_transform
    extract_loader = DataLoader(
        extract_dataset,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=False,
    )

    train(model, train_loader, test_loader_unlabelled, extract_loader, args)


if __name__ == "__main__":
    main()
