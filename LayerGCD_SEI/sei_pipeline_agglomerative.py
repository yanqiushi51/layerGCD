#!/usr/bin/env python3
"""
RF Signal Deep Clustering System (Full Monitor Version)
Features:
  1. Model: Student (Feature Extractor) trained via Decoupled SSL.
  2. Eval: K-Means on Test Set (Purity/NMI/ARI).
  3. Monitor: 3-Panel Viz (GT/Match/Selection) per iteration.
  4. Report: Detailed Pseudo-label accuracy analysis for selected samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import os
import warnings
import argparse
import json

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
for cache_dir in ("NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    Path(os.environ[cache_dir]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import copy
import re
from agglomerative_discovery import dynamic_cluster_discovery, estimate_known_calibrated_threshold

warnings.filterwarnings('ignore')

# ==============================================================================
# [Part 0] Config
# ==============================================================================

# Seed is applied in main() after parsing --seed argument

RESULT_DIR = "results_deep_cluster_monitor"
os.makedirs(RESULT_DIR, exist_ok=True)

class Config:
    def __init__(self):
        # Data
        self.data_dir = "data/LFM_dataset/data_noise_30"
        self.known_classes_initial = [1, 2, 3, 4, 5, 6, 7] 
        self.unknown_classes_real = [8, 9, 10] 
        self.signal_length = 200
        self.total_classes_eval = 10 
        self.class_to_label = {}
        self.use_fractal = True
        self.enable_pretrain = True
        self.enable_visualization = True
        self.variant = "full"
        
        # DS-MFAN Params
        self.temporal_dim = 64
        self.fractal_dim = 16
        self.fusion_dim = 128
        self.n_heads = 4
        self.n_layers = 2
        self.fractal_scales = [3, 4, 6, 8, 12, 16, 24, 32] 
        self.feature_dim = 128
        
        # Training
        self.batch_size = 64
        # When launched via run_paper_experiments.py, CUDA_VISIBLE_DEVICES is set
        # so only one GPU is visible → must use cuda:0.
        # When running standalone, default to cuda:1 to avoid conflicts.
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        
        self.epochs_teacher = 15
        self.epochs_pretrain = 25
        self.max_iterations = 1
        self.epochs_per_iter = 10
        self.seed = 42
        
        # Discovery
        self.merge_threshold = None
        self.active_merge_threshold = None
        self.threshold_info = {}
        self.selection_ratio = 0.3      
        self.uncertainty_weight = 1.0
        self.match_threshold = 0.001 
        self.umap_dim_clustering = 10
        
        # 放大最大支持类别，避免丢弃有价值的伪标签聚集簇
        self.max_total_classes = 100
        
        # Loss
        self.smooth_kernel_size = 5
        self.smooth_sigma = 1.0
        self.lambda_ent = 0.5      
        self.ent_warmup_epochs = 5 

# ==============================================================================
# [Part 1] Data Loading
# ==============================================================================

def list_available_class_indices(data_dir):
    class_indices = []
    for path in Path(data_dir).glob("*.mat"):
        match = re.fullmatch(r"data_?(\d+)\.mat", path.name)
        if match:
            class_indices.append(int(match.group(1)))
    return sorted(set(class_indices))

def resolve_class_file(data_dir, cls_idx):
    data_dir = Path(data_dir)
    candidates = [data_dir / f"data{cls_idx}.mat", data_dir / f"data_{cls_idx}.mat"]
    for path in candidates:
        if path.exists():
            return path
    return None

def load_data_by_split(config, class_indices, split='train'):
    all_data, all_labels = [], []
    for cls_idx in class_indices:
        path = resolve_class_file(config.data_dir, cls_idx)
        if path is None:
            print(f"Missing class file for class {cls_idx} in {config.data_dir}")
            continue
        try:
            mat = loadmat(str(path))
            valid_keys = [k for k in mat.keys() if not k.startswith('__')]
            if not valid_keys:
                continue
            data_key = 'all_data' if 'all_data' in valid_keys else valid_keys[0]
            mat = np.asarray(mat[data_key], dtype=np.float32)
            processed = []
            for j in range(mat.shape[0]):
                sig = np.abs(mat[j, :])
                if np.max(sig) > 0: sig = sig / np.max(sig)
                if len(sig) > config.signal_length:
                    sig = sig[:config.signal_length]
                elif len(sig) < config.signal_length:
                    sig = np.pad(sig, (0, config.signal_length - len(sig)), 'constant')
                processed.append(sig)
            data_np = np.array(processed)
            if split == 'train': selected = data_np[:4000]
            elif split == 'test': selected = data_np[4000:5000]
            else: continue
            if len(selected) > 0:
                all_data.append(selected)
                all_labels.extend([config.class_to_label[cls_idx]] * len(selected))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if len(all_data) == 0: return None, None
    return np.concatenate(all_data).astype(np.float32), np.array(all_labels, dtype=np.int64)

# ==============================================================================
# [Part 2] Helpers: Hungarian & Viz
# ==============================================================================

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def visualize_iter_3panel(feats_2d, labels_gmm, mask_selected, true_labels, 
                         final_label_map, iter_num, config):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6)) 
    
    # --- Panel 1: Ground Truth ---
    ax = axes[0]
    unique_true = np.unique(true_labels)
    cmap = plt.get_cmap('tab20', len(unique_true) + 1)
    known_set = set(config.known_classes_initial) 
    
    for i, lbl in enumerate(unique_true):
        idx = true_labels == lbl
        # Physical ID = lbl + 1
        phy_id = lbl + 1
        is_known = phy_id in known_set
        marker = 'o' if is_known else '^'
        prefix = "K" if is_known else "U"
        ax.scatter(feats_2d[idx, 0], feats_2d[idx, 1], s=15, alpha=0.7, marker=marker, 
                   color=cmap(i), label=f"{prefix}-{phy_id}")
    
    ax.set_title(f"Iter {iter_num}: Ground Truth")
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), title="Classes", fontsize='small', frameon=True)

    # --- Panel 2: Cluster Matching (God Mode) ---
    ax = axes[1]
    unique_gmm = np.unique(labels_gmm)
    cmap_gmm = plt.get_cmap('viridis', len(unique_gmm))
    
    # Helper: Compute cluster purity for legend
    D = max(true_labels.max(), labels_gmm.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels_gmm)): w[true_labels[i], labels_gmm[i]] += 1
    r_ind, c_ind = linear_sum_assignment(w.max() - w)
    clus_to_true = {c:r for r,c in zip(r_ind, c_ind)}

    for i, lbl in enumerate(unique_gmm):
        idx = labels_gmm == lbl
        
        # Logic 1: What did we assign it?
        if lbl in final_label_map:
            assigned_id, _ = final_label_map[lbl]
            assign_txt = f"{assigned_id+1}"
        else:
            assign_txt = "Ign"
            
        # Logic 2: What is it really? (God Mode)
        true_id = clus_to_true.get(lbl, -1) + 1
        
        label_text = f"C{lbl} -> P:{assign_txt} (True:{true_id})"
        alpha = 0.6
        size = 15
            
        ax.scatter(feats_2d[idx, 0], feats_2d[idx, 1], s=size, alpha=alpha, color=cmap_gmm(i), label=label_text)
        
    ax.set_title("Cluster Matching (Viz)")
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='x-small', title="Clus -> Pseudo(True)")

    # --- Panel 3: Selection ---
    ax = axes[2]
    ax.scatter(feats_2d[~mask_selected, 0], feats_2d[~mask_selected, 1], s=10, c='lightgray', alpha=0.2, label='Ignored')
    ax.scatter(feats_2d[mask_selected, 0], feats_2d[mask_selected, 1], s=15, c='red', alpha=0.8, label='Selected')
    ax.set_title(f"Selection (Count: {mask_selected.sum()})")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"viz_iter_{iter_num:02d}.png"), bbox_inches='tight')
    plt.close()

# ==============================================================================
# [Part 3] Loss Functions
# ==============================================================================

class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=100):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
    def forward(self, x, y):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        mu = torch.empty(x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).to(x.device)
        nu = torch.empty(y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).to(x.device)
        P = self._sinkhorn(C, mu, nu)
        cost = torch.sum(P * C)
        return cost
    def _cost_matrix(self, x, y):
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        cosine = torch.mm(x_norm, y_norm.t())
        cost = 1 - cosine
        return cost
    def _sinkhorn(self, C, mu, nu):
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(mu)
        for _ in range(self.max_iter):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)
        P = torch.diag(u) @ K @ torch.diag(v)
        return P

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, features):
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.t
        mask = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
        sim.masked_fill_(mask, -9e15)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(features.device)
        return F.cross_entropy(sim, labels)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        sim = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(features.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)
        mask_sum = mask.sum(dim=1)
        loss = - (mask * log_prob).sum(dim=1) / (mask_sum + 1e-7)
        loss = loss[mask_sum > 0].mean()
        return loss if not torch.isnan(loss) else torch.tensor(0.0).to(device)

def gaussian_smooth(x, kernel_size=5, sigma=1.0):
    if x.dim() == 2: x_in = x.unsqueeze(1)
    else: x_in = x
    b, c, l = x_in.size()
    k_half = kernel_size // 2
    x_padded = F.pad(x_in, (k_half, k_half), mode='reflect')
    x_grid = torch.arange(-k_half, k_half + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-(x_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).repeat(c, 1, 1)
    x_smoothed = F.conv1d(x_padded, kernel, groups=c)
    if x.dim() == 2: return x_smoothed.squeeze(1)
    return x_smoothed

# ==============================================================================
# [Part 4] Model Architecture (DS-MFAN)
# ==============================================================================

class DifferentiableFractalLayer(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = scales
        X_mat = np.vstack([np.log(scales), np.ones(len(scales))]).T
        P = np.linalg.pinv(X_mat)
        self.register_buffer('regression_kernel', torch.from_numpy(P[0, :]).float())
    def get_local_fluctuation(self, x, k):
        pad_len = k // 2
        pad_l, pad_r = (pad_len, k - 1 - pad_len) if k % 2 == 0 else (pad_len, pad_len)
        x_padded = F.pad(x, (pad_l, pad_r), mode='replicate')
        trend = F.avg_pool1d(x_padded, kernel_size=k, stride=1)
        residual = x - trend
        res_padded = F.pad(torch.abs(residual), (pad_l, pad_r), mode='replicate')
        return F.avg_pool1d(res_padded, kernel_size=k, stride=1)
    def forward(self, x):
        log_Fs = [torch.log(self.get_local_fluctuation(x, s) + 1e-6) for s in self.scales]
        stack_log_F = torch.stack(log_Fs, dim=1).squeeze(2)
        kernel = self.regression_kernel.view(1, -1, 1)
        return torch.sum(stack_log_F * kernel, dim=1, keepdim=True)

class GatedFusionLayer(nn.Module):
    def __init__(self, dim_temp, dim_frac):
        super().__init__()
        self.gate_temp = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim_temp, dim_temp//4, 1), nn.ReLU(), nn.Conv1d(dim_temp//4, dim_temp, 1), nn.Sigmoid())
        self.gate_frac = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim_frac, dim_frac//2, 1), nn.ReLU(), nn.Conv1d(dim_frac//2, dim_frac, 1), nn.Sigmoid())
    def forward(self, x_t, x_f):
        return torch.cat([x_t * self.gate_temp(x_t), x_f * self.gate_frac(x_f)], dim=1)

class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm1': nn.LayerNorm(d_model), 'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, d_model)),
                'dropout': nn.Dropout(0.1)
            }) for _ in range(num_layers)])
    def forward(self, src):
        output, attn_maps = src, []
        for layer in self.layers:
            src2, weights = layer['self_attn'](output, output, output, need_weights=True)
            attn_maps.append(weights)
            output = layer['norm1'](output + layer['dropout'](src2))
            src2 = layer['ffn'](output)
            output = layer['norm2'](output + layer['dropout'](src2))
        return output, attn_maps

class DSMFAN_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_fractal = config.use_fractal
        self.shape_enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, config.temporal_dim, 5, padding=2), nn.BatchNorm1d(config.temporal_dim), nn.ReLU())
        if self.use_fractal:
            self.frac_ext = DifferentiableFractalLayer(config.fractal_scales)
            self.frac_emb = nn.Sequential(nn.Conv1d(1, config.fractal_dim, 3, padding=1), nn.ReLU())
            self.fusion = GatedFusionLayer(config.temporal_dim, config.fractal_dim)
            self.proj = nn.Conv1d(config.temporal_dim + config.fractal_dim, config.fusion_dim, 1)
        else:
            self.temporal_proj = nn.Conv1d(config.temporal_dim, config.fusion_dim, 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.signal_length, config.fusion_dim))
        self.transformer = TransformerEncoderWithAttn(config.fusion_dim, config.n_heads, config.n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x_s = self.shape_enc(x)
        if self.use_fractal:
            x_f = self.frac_emb(self.frac_ext(x))
            fused = self.proj(self.fusion(x_s, x_f))
        else:
            fused = self.temporal_proj(x_s)
        x_seq = fused.permute(0, 2, 1) + self.pos_emb
        x_seq, attns = self.transformer(x_seq)
        return self.pool(x_seq.permute(0, 2, 1)).flatten(1), attns

class StudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DSMFAN_Backbone(config)
        self.projector = nn.Sequential(nn.Linear(config.fusion_dim, config.fusion_dim), nn.ReLU(), nn.Linear(config.fusion_dim, 128))
        self.decoder = nn.Sequential(nn.Linear(config.fusion_dim, 256), nn.ReLU(), nn.Linear(256, config.signal_length))
    def forward(self, x):
        feats, attns = self.backbone(x)
        return feats, attns, self.projector(feats), self.decoder(feats)

# ==============================================================================
# [Part 5] Training & Discovery Logic
# ==============================================================================

def train_phase_1_student_pretrain(student, loader_all, device, epochs):
    print(f"🌱 [Phase 1] Student Pre-training...")
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    for ep in range(epochs):
        for d, _ in loader_all:
            d = d.to(device)
            _, _, _, recon = student(d)
            loss = crit(recon, d.squeeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return student

def train_phase_2_loop(student, loader_k, loader_u, device, epochs, config):
    print(f"🔄 [Phase 2] Student Loop...")
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=0.0005)
    crit_sim = SimCLRLoss(0.07)
    crit_supcon = SupConLoss(0.07)
    
    for ep in range(epochs):
        iter_k = iter(loader_k)
        iter_u = iter(loader_u) if loader_u else None
        loops = min(len(loader_k), len(loader_u)) if loader_u else len(loader_k)
        
        for _ in range(loops):
            try:
                d_k, t_k = next(iter_k)
                if iter_u: d_u, _ = next(iter_u)
                else: d_u = None
            except: break
            
            d_k, t_k = d_k.to(device), t_k.to(device)
            if d_u is not None: d_u = d_u.to(device)
            
            # SupCon (已知类)
            _, _, proj_k, _ = student(d_k)
            d_k_aug = gaussian_smooth(d_k, config.smooth_kernel_size)
            _, _, proj_k_aug, _ = student(d_k_aug)
            
            loss_supcon = crit_supcon(torch.cat([proj_k, proj_k_aug]), torch.cat([t_k, t_k]))
            loss_total = 1.0 * loss_supcon
            
            # SimCLR
            if d_u is not None:
                d_u_aug = gaussian_smooth(d_u, config.smooth_kernel_size)
                _, _, proj_u, _ = student(d_u)
                _, _, proj_u_aug, _ = student(d_u_aug)
                loss_simclr = crit_sim(torch.cat([proj_u, proj_u_aug]))
                
                loss_total += 0.5 * loss_simclr
            
            opt.zero_grad()
            loss_total.backward()
            opt.step()
    return student

def discovery_and_match(student, loader_u, known_protos, current_max_id, config, iteration):
    print(f"   🔍 Discovery & Visualization...")
    student.eval()
    feats, data_store, true_lbls = [], [], []
    with torch.no_grad():
        for d, t in loader_u:
            d = d.to(config.device)
            f, _, _, _ = student(d)
            feats.append(f.cpu().numpy())
            data_store.append(d.cpu().numpy())
            true_lbls.append(t.numpy())
    if not feats:
        return (
            np.empty((0, config.signal_length), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, config.signal_length), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            {'selected_purity': None, 'promoted_count': 0},
        )
    feats = np.concatenate(feats)
    data_store = np.concatenate(data_store)
    true_lbls = np.concatenate(true_lbls)
    
    # Clustering
    feats_2d = None
    if config.enable_visualization:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        feats_2d = reducer.fit_transform(feats)
        
    feats_norm = F.normalize(torch.tensor(feats), dim=1).numpy()
    labels, centers, num_discovered, used_tau = dynamic_cluster_discovery(
        feats_norm,
        distance_threshold=config.active_merge_threshold,
        auto_threshold=(config.active_merge_threshold is None),
    )
    print(f"   Agglomerative Discovery: {num_discovered} clusters (tau={used_tau:.4f})")
    
    final_map = {}
    known_labels = sorted(list(known_protos.keys()))
    assigned_known = set()
    
    # Match Known
    for c_id, center in centers.items():
        sims = {k: np.dot(center, known_protos[k].cpu().numpy()) for k in known_labels}
        best_k = max(sims, key=sims.get)
        if sims[best_k] > config.match_threshold and best_k not in assigned_known:
            final_map[c_id] = (best_k, 'Old')
            assigned_known.add(best_k)
    
    # Assign New
    unmapped = [c for c in centers if c not in final_map]
    unmapped.sort(key=lambda x: (labels==x).sum(), reverse=True)
    
    max_known = max(known_labels) if known_labels else -1
    curr_new = max(max_known + 1, len(known_labels))
    
    for c_id in unmapped:
        if curr_new > config.max_total_classes - 1: 
            print(f"⚠️ [Warning] Discarding cluster {c_id} due to max_total_classes={config.max_total_classes} limit.")
            break
        final_map[c_id] = (curr_new, 'New')
        curr_new += 1
        
    mask = np.zeros(len(feats), dtype=bool)
    pseudo_y = np.full(len(feats), -1)
    
    for c_id, (k_id, _) in final_map.items():
        idx = labels == c_id
        scores = np.dot(feats_norm[idx], centers[c_id])
        k_sel = int(len(scores) * config.selection_ratio)
        if k_sel > 0:
            sel_local = np.argsort(scores)[-k_sel:]
            glob_idx = np.where(idx)[0][sel_local]
            mask[glob_idx] = True
            pseudo_y[glob_idx] = k_id
            
    # [Analysis] Print Pseudo-Label Purity for Selected Samples
    sel_y_true = true_lbls[mask]
    sel_y_pseudo = pseudo_y[mask]
    selected_purity = None
    if len(sel_y_true) > 0:
        # Hungarian Purity Check
        D = max(sel_y_true.max(), sel_y_pseudo.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(len(sel_y_true)): w[sel_y_true[i], sel_y_pseudo[i]] += 1
        r_ind, c_ind = linear_sum_assignment(w.max() - w)
        selected_purity = w[r_ind, c_ind].sum() / len(sel_y_true)
        print(f"   📊 Selected Batch Purity (Hungarian): {selected_purity:.4f}")
        
        # Detailed Breakdown
        print(f"   📋 Breakdown (Pseudo -> Major True):")
        for pid in np.unique(sel_y_pseudo):
            sub_true = sel_y_true[sel_y_pseudo == pid]
            counts = np.bincount(sub_true)
            major = counts.argmax()
            prop = counts[major] / len(sub_true)
            print(f"      Pseudo {pid:<2} -> True {major:<2} ({prop:.1%})")

    if config.enable_visualization and feats_2d is not None:
        visualize_iter_3panel(feats_2d, labels, mask, true_lbls, final_map, iteration, config)
    
    stats = {
        'selected_purity': float(selected_purity) if selected_purity is not None else None,
        'promoted_count': int(mask.sum()),
    }
    return data_store[mask], pseudo_y[mask], true_lbls[mask], data_store[~mask], true_lbls[~mask], stats

def evaluate_clustering_performance(model, loader, device, step_name, config, prototypes=None, is_final=False):
    from sklearn.metrics import roc_auc_score
    model.eval()
    feats_all, targets_all = [], []
    with torch.no_grad():
        for d, t in loader:
            d = d.to(device)
            f, _, _, _ = model(d)
            f = F.normalize(f, dim=1)
            feats_all.append(f.cpu().numpy())
            targets_all.append(t.numpy())
    feats = np.concatenate(feats_all)
    y_true = np.concatenate(targets_all)
    
    y_pred, _, num_pred_classes, used_tau = dynamic_cluster_discovery(
        feats,
        distance_threshold=config.active_merge_threshold,
        auto_threshold=(config.active_merge_threshold is None),
    )
    print(f"   Eval Auto-Discovered: {num_pred_classes} clusters (tau={used_tau:.4f})")
    
    if is_final and config.enable_visualization:
        import matplotlib.pyplot as plt
        from agglomerative_discovery import estimate_optimal_threshold
        # 只采样，避免全量 O(N²)
        _, hist_info = estimate_optimal_threshold(feats, n_sample_pairs=100000)
        bin_centers = hist_info['bin_centers']
        hist_smooth = hist_info['hist_smooth']
        valley_x = bin_centers[hist_info['valley_idx']]
        peak_x   = bin_centers[hist_info['first_peak_idx']]
        
        plt.figure(figsize=(11, 6))
        plt.plot(bin_centers, hist_info['hist'], alpha=0.3, color='gray', linewidth=1, label='原始分布')
        plt.plot(bin_centers, hist_smooth, color='black', linewidth=2, label='平滑后分布')
        plt.axvline(x=peak_x,   color='red',   linestyle='--', linewidth=1.5,
                    label=f'第一峰 (intra 主体) x={peak_x:.3f}')
        plt.axvline(x=valley_x, color='green', linestyle='--', linewidth=2.0,
                    label=f'波谷 → 推荐 τ={valley_x:.3f}')
        plt.axvline(x=used_tau, color='orange', linestyle='-', linewidth=1.5,
                    label=f'本次实际使用 τ={used_tau:.3f}')
        plt.title('Cosine Distance Distribution (Unsupervised Auto-Threshold)')
        plt.xlabel('Cosine Distance (1 - cosine similarity)')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(RESULT_DIR, "final_trained_similarity_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Trained cosine distance distribution → {save_path}")
        print(f"   💡 推荐 tau={valley_x:.4f}（对应相似度门限 {1-valley_x:.4f}），本次使用 {used_tau:.4f}")
    
    D = max(y_pred.max() if len(y_pred) > 0 else 0, y_true.max() if len(y_true) > 0 else 0) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size): w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    # All ACC
    all_acc = w[row_ind, col_ind].sum() / y_pred.size
    
    # Map known_classes_initial to label space for correct comparison with y_true
    known_labels = set()
    for c in config.known_classes_initial:
        if c in config.class_to_label:
            known_labels.add(config.class_to_label[c])
    
    # Old ACC / New ACC
    is_known = np.isin(y_true, list(known_labels))
    is_unknown = ~is_known
    
    old_acc = 0.0
    if is_known.sum() > 0:
        w_old = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            if is_known[i]:
                w_old[y_pred[i], y_true[i]] += 1
        old_acc = w_old[row_ind, col_ind].sum() / is_known.sum()
        
    new_acc = 0.0
    if is_unknown.sum() > 0:
        w_new = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            if is_unknown[i]:
                w_new[y_pred[i], y_true[i]] += 1
        new_acc = w_new[row_ind, col_ind].sum() / is_unknown.sum()

    # OS-ACC (K+1 accuracy)
    clus_map = {r: c for r, c in zip(row_ind, col_ind)}
    UNKNOWN_LBL = -1
    y_pred_k1 = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        mapped_true = clus_map.get(y_pred[i], UNKNOWN_LBL)
        if mapped_true in known_labels:
            y_pred_k1[i] = mapped_true
        else:
            y_pred_k1[i] = UNKNOWN_LBL
            
    y_true_k1 = np.where(is_known, y_true, UNKNOWN_LBL)
    os_acc = np.sum(y_pred_k1 == y_true_k1) / len(y_true)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # AUROC Calculation (prototypes are keyed by label index)
    auroc = 0.0
    if prototypes is not None and is_known.sum() > 0 and is_unknown.sum() > 0:
        # compute max cosine similarity to any known prototype
        protos_list = []
        for lbl in known_labels:
            if lbl in prototypes:
                p = prototypes[lbl]
                if isinstance(p, torch.Tensor):
                    p = p.cpu().numpy()
                protos_list.append(p)
        if len(protos_list) > 0:
            protos_mat = np.stack(protos_list) # (K, feature_dim)
            sims = np.dot(feats, protos_mat.T) # (N, K)
            max_sims = np.max(sims, axis=1) # (N,)
            auroc = roc_auc_score(is_known.astype(int), max_sims)
    
    print(f"\n📈 [{step_name}] Test Set Metrics:")
    print(f"   Overall: All-ACC={all_acc:.4f} | OS-ACC={os_acc:.4f} | NMI={nmi:.4f} | ARI={ari:.4f}")
    print(f"   Old ACC={old_acc:.4f} | New ACC={new_acc:.4f} | AUROC={auroc:.4f}")
    
    # Detailed Report
    print("-" * 65)
    print(f"   {'Cluster':<8} | {'Mapped True':<12} | {'Purity':<8} | {'Composition'}")
    print("-" * 65)
    for c in range(num_pred_classes):
        idx = y_pred == c
        if idx.sum() == 0: continue
        sub_y = y_true[idx]
        counts = np.bincount(sub_y)
        major = counts.argmax()
        pur = counts[major] / len(sub_y)
        
        mapped = clus_map.get(c, -1) + 1
        # Top 3
        top3 = np.argsort(counts)[-3:][::-1]
        comp = ", ".join([f"C{i+1}:{counts[i]}" for i in top3 if counts[i]>0])
        print(f"   C{c:<7} | Class {mapped:<6} | {pur:.1%}  | {comp}")
    print("-" * 65)
    return {
        'acc': float(all_acc),
        'os_acc': float(os_acc),
        'old_acc': float(old_acc),
        'new_acc': float(new_acc),
        'auroc': float(auroc),
        'nmi': float(nmi),
        'ari': float(ari),
        'num_pred_classes': int(num_pred_classes),
        'used_tau': float(used_tau),
    }


def calibrate_threshold_from_known(model, loader, device):
    model.eval()
    feats_all, labels_all = [], []
    with torch.no_grad():
        for d, t in loader:
            d = d.to(device)
            f, _, _, _ = model(d)
            f = F.normalize(f, dim=1)
            feats_all.append(f.cpu().numpy())
            labels_all.append(t.numpy())
    feats = np.concatenate(feats_all)
    labels = np.concatenate(labels_all)
    return estimate_known_calibrated_threshold(feats, labels)

# ==============================================================================
# [Part 7] Main
# ==============================================================================

def parse_merge_threshold(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"auto", "none", "null"}:
        return None
    return float(text)


def main():
    parser = argparse.ArgumentParser(description="LFM SEI Pipeline")
    parser.add_argument('--data_dir', type=str, default="data/LFM_dataset/data_noise_30", help="Path to LFM dataset")
    parser.add_argument('--known_count', type=int, default=7, help="Number of known classes")
    parser.add_argument('--variant', type=str, default="full", choices=['full', 'no_fractal', 'no_recon', 'pure_base'], help="Experiment variant")
    parser.add_argument('--metrics_out', type=str, default="", help="Optional path to write structured metrics JSON")
    parser.add_argument('--disable_visualization', action='store_true', help="Skip UMAP and visualization image generation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--max_iterations', type=int, default=1, help="Number of self-training iterations")
    parser.add_argument('--run_idx', type=int, default=0, help="ID for multiple runs under the same network seed")
    parser.add_argument('--merge_threshold', type=parse_merge_threshold, default=None,
                        help="Cosine-distance threshold for average-linkage HAC; use 'auto' for valley estimation.")
    parser.add_argument('--epochs_pretrain', type=int, default=None,
                        help="Override reconstruction pre-training epochs.")
    parser.add_argument('--epochs_per_iter', type=int, default=None,
                        help="Override contrastive epochs per discovery iteration.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Override training/evaluation batch size.")
    parser.add_argument('--selection_ratio', type=float, default=None,
                        help="Fraction of high-confidence samples promoted from each discovered cluster.")
    args = parser.parse_args()

    # Apply seed globally as early as possible
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    config = Config()
    config.data_dir = args.data_dir
    config.variant = args.variant
    config.use_fractal = args.variant in {'full', 'no_recon'}
    config.enable_pretrain = args.variant in {'full', 'no_fractal'}
    config.enable_visualization = not args.disable_visualization
    # None → 触发自动阈值估算；指定浮点数 → 手动覆盖
    config.merge_threshold = args.merge_threshold
    if args.epochs_pretrain is not None:
        config.epochs_pretrain = args.epochs_pretrain
    if args.epochs_per_iter is not None:
        config.epochs_per_iter = args.epochs_per_iter
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.selection_ratio is not None:
        config.selection_ratio = args.selection_ratio

    available_classes = list_available_class_indices(config.data_dir)
    if len(available_classes) < 2:
        raise ValueError(f"Need at least 2 classes in {config.data_dir}, found {len(available_classes)}")
    if args.known_count < 1 or args.known_count >= len(available_classes):
        raise ValueError(f"--known_count must be in [1, {len(available_classes) - 1}], got {args.known_count}")

    config.class_to_label = {cls_idx: label_idx for label_idx, cls_idx in enumerate(available_classes)}
    
    # 结合 network seed 和 run_idx 保证同一 seed 下可以跑多次不同的随机选取
    rng = np.random.RandomState(args.seed + args.run_idx * 2026)
    shuffled_classes = available_classes.copy()
    rng.shuffle(shuffled_classes)
    
    config.known_classes_initial = shuffled_classes[:args.known_count]
    config.unknown_classes_real = shuffled_classes[args.known_count:]
    config.total_classes_eval = len(available_classes)
    config.max_total_classes = len(available_classes)
    config.max_iterations = args.max_iterations
    config.seed = args.seed

    print(f"🚀 Starting Deep Clustering System (Full Monitor)")
    print(f"   Dataset: {config.data_dir}")
    print(f"   Variant: {config.variant}")
    print(f"   Known/Unknown: {len(config.known_classes_initial)} + {len(config.unknown_classes_real)}")
    print(f"   Device: {config.device}")
    
    x_k_all, y_k_all = load_data_by_split(config, config.known_classes_initial, 'train')
    x_u_all, y_u_all = load_data_by_split(config, config.unknown_classes_real, 'train')
    x_test, y_test = load_data_by_split(config, config.known_classes_initial + config.unknown_classes_real, 'test')
    
    # Init Split
    num_known = len(x_k_all)
    idx = np.arange(num_known)
    np.random.shuffle(idx)
    split = int(num_known * 0.5)
    
    x_k, y_k = x_k_all[idx[:split]], y_k_all[idx[:split]]
    x_u = np.concatenate([x_k_all[idx[split:]], x_u_all])
    y_u = np.concatenate([y_k_all[idx[split:]], y_u_all])
    
    curr_x_k, curr_y_k = [x_k], [y_k]
    curr_x_u, curr_y_u = x_u, y_u
    
    student = StudentModel(config).to(config.device)
    
    loader_k = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_k), torch.LongTensor(y_k)), batch_size=config.batch_size, shuffle=True)
    loader_calib = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_k), torch.LongTensor(y_k)), batch_size=config.batch_size, shuffle=False)
    loader_all = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(np.concatenate([x_k, x_u])), torch.LongTensor(np.concatenate([y_k, y_u]))), batch_size=config.batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test)), batch_size=config.batch_size, shuffle=False)
    
    if config.enable_pretrain:
        student = train_phase_1_student_pretrain(student, loader_all, config.device, config.epochs_pretrain)
    else:
        print(f"⏭️ [Phase 1] Skipped reconstruction pre-training for variant={config.variant}")
    
    prototypes = None
    history = []
    
    for i in range(config.max_iterations):
        print(f"\n=== 🔄 Iteration {i+1} ===")
        
        x_k_f = np.concatenate(curr_x_k)
        y_k_f = np.concatenate(curr_y_k)
        loader_k = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(x_k_f), torch.LongTensor(y_k_f)), batch_size=config.batch_size, shuffle=True, drop_last=True)
        loader_u = None
        if len(curr_x_u) > config.batch_size:
            loader_u = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(curr_x_u), torch.LongTensor(curr_y_u)), batch_size=config.batch_size, shuffle=True, drop_last=True)
        student = train_phase_2_loop(student, loader_k, loader_u, config.device, config.epochs_per_iter, config)

        if config.merge_threshold is None:
            config.active_merge_threshold, config.threshold_info = calibrate_threshold_from_known(student, loader_calib, config.device)
            print(
                f"   Known-calibrated HAC tau={config.active_merge_threshold:.4f} "
                f"(intra_q={config.threshold_info.get('intra_q', 0):.4f}, "
                f"inter_q={config.threshold_info.get('inter_q', 0):.4f})"
            )
        else:
            config.active_merge_threshold = config.merge_threshold
            config.threshold_info = {}
        
        student.eval()
        feat_sum, counts = {}, {}
        with torch.no_grad():
            for d, t in loader_k:
                d = d.to(config.device)
                f, _, _, _ = student(d) 
                f = F.normalize(f, dim=1)
                for j in range(len(t)):
                    l = t[j].item()
                    feat_sum[l] = feat_sum.get(l, 0) + f[j]
                    counts[l] = counts.get(l, 0) + 1
        prototypes = {k: v/counts[k] for k, v in feat_sum.items()}
        
        discovery_stats = {'selected_purity': None, 'promoted_count': 0}
        if loader_u:
            x_sel, y_ps, y_tr, x_rm, y_tr_rm, discovery_stats = discovery_and_match(student, loader_u, prototypes, 0, config, i+1)
            if len(x_sel) > 0:
                print(f"✅ Promoted {len(x_sel)} samples.")
                curr_x_k.append(x_sel)
                curr_y_k.append(y_ps)
                curr_x_u, curr_y_u = x_rm, y_tr_rm
        
        is_final_iter = (i == config.max_iterations - 1)
        eval_metrics = evaluate_clustering_performance(student, loader_test, config.device, f"Iter {i+1}", config, prototypes, is_final=is_final_iter)
        history.append({
            'iteration': i + 1,
            **eval_metrics,
            **discovery_stats,
        })

    print("\n🎉 Process Finished.")
    best_iter = max(history, key=lambda item: (item['acc'], item['nmi'], item['ari']))
    final_iter = history[-1]
    results_payload = {
        'data_dir': config.data_dir,
        'dataset_name': Path(config.data_dir).name,
        'variant': config.variant,
        'known_count': len(config.known_classes_initial),
        'unknown_count': len(config.unknown_classes_real),
        'known_classes': config.known_classes_initial,
        'unknown_classes': config.unknown_classes_real,
        'device': config.device,
        'cuda_available': torch.cuda.is_available(),
        'merge_threshold_mode': 'known_calibrated' if config.merge_threshold is None else 'manual',
        'threshold_info': config.threshold_info,
        'iterations': history,
        'best_iter': best_iter,
        'final_iter': final_iter,
    }
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open('w', encoding='utf-8') as f:
            json.dump(results_payload, f, indent=2)
        print(f"📝 Metrics written to {metrics_path}")
    return results_payload

if __name__ == "__main__":
    main()
