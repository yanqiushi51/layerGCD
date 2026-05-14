"""
Hierarchical clustering module for LayerGCD.

Builds a multi-level clustering tree where each level uses features
from a different DINO layer, creating a natural semantic hierarchy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm


class HierarchicalClusterTree:
    """
    Builds and maintains a multi-level clustering hierarchy.
    
    Unlike SelEx (which uses dimension slicing on the same features), 
    this uses features from different DINO layers for different hierarchy levels:
    - Deepest DINO layer (e.g., 12) → finest clustering (full K classes)
    - Shallower layers (e.g., 11, 9, 7) → progressively coarser clustering
    """
    
    def __init__(self, extract_layers, n_labeled, n_unlabeled, min_classes=8,
                 use_label_anchors=True, label_anchor_weight=10.0,
                 seeded_kmeans_iters=10):
        """
        Args:
            extract_layers: Sorted list of DINO block indices
            n_labeled: Number of labeled (known) classes
            n_unlabeled: Number of unlabeled (novel) classes
            min_classes: Minimum clusters at coarsest level
        """
        self.extract_layers = extract_layers
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.min_classes = min_classes
        self.use_label_anchors = use_label_anchors
        self.label_anchor_weight = label_anchor_weight
        self.seeded_kmeans_iters = seeded_kmeans_iters
        
        # Compute number of clusters per level
        # Deepest layer → K, then halve going to shallower layers
        total_classes = n_labeled + n_unlabeled
        self.n_clusters_per_level = {}
        for i, layer_idx in enumerate(reversed(extract_layers)):
            n_cls = min(total_classes, max(total_classes // (2 ** i), min_classes))
            self.n_clusters_per_level[layer_idx] = n_cls
        
        # Storage for prototypes and pseudo-labels
        self.prototypes = {}      # layer_idx → [n_clusters, feat_dim]
        self.pseudo_labels = {}   # layer_idx → [n_samples]
        self.cluster_radii = {}   # layer_idx → [n_clusters]
        self.index_to_position = {}
        self.fine_to_level = {}   # layer_idx → [n_fine_clusters], fine slot to level cluster
    
    @torch.no_grad()
    def build_hierarchy(self, model, dataloader, device='cuda'):
        """
        Build the full hierarchy tree.
        
        Process:
        1. Extract features from all layers for all samples
        2. At the deepest layer: run KMeans with full K clusters
        3. At each shallower layer: cluster the previous level's prototypes
           to get coarser groupings, then reassign samples using that layer's features
        
        Args:
            model: MultiLayerDINO model
            dataloader: DataLoader for the training set
            device: torch device
        """
        model.eval()
        
        # Step 1: Extract features from all layers
        all_features = {layer: [] for layer in self.extract_layers}
        all_indices = []
        all_labels = []
        all_masks = []  # labeled/unlabeled mask
        
        for batch in tqdm(dataloader, desc='Extracting features for hierarchy'):
            # Some loaders return (img, label, idx), some return (img, label, idx, mask)
            if len(batch) == 4:
                images, labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]
            else:
                images, labels, uq_idxs = batch
                mask_lab = torch.zeros_like(labels).bool()
                
            images = images[0].to(device) if isinstance(images, list) else images.to(device)
            
            layer_feats = model(images, return_all_layers=True)
            
            for layer_idx, feat in layer_feats.items():
                all_features[layer_idx].append(
                    F.normalize(feat, dim=-1).cpu()
                )
            all_indices.append(uq_idxs)
            all_labels.append(labels)
            all_masks.append(mask_lab)
        
        # Concatenate
        for layer_idx in self.extract_layers:
            all_features[layer_idx] = torch.cat(all_features[layer_idx], dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0).bool()

        # Sort by index to ensure consistent ordering
        sort_order = torch.argsort(all_indices)
        all_indices = all_indices[sort_order]
        for layer_idx in self.extract_layers:
            all_features[layer_idx] = all_features[layer_idx][sort_order]
        all_labels = all_labels[sort_order]
        all_masks = all_masks[sort_order]
        self.index_to_position = {
            int(sample_idx): pos for pos, sample_idx in enumerate(all_indices.tolist())
        }
        
        # Step 2: SelEx-style bottom-up hierarchy construction
        # Use one shared embedding space (deepest layer features) to build all levels.
        deepest_layer = max(self.extract_layers)
        base_feats_np = all_features[deepest_layer].numpy()
        
        n_clusters = self.n_clusters_per_level[deepest_layer]
        if self.use_label_anchors and self.n_labeled > 0 and all_masks.any():
            print(
                f'  Level {deepest_layer}: seeded KMeans with '
                f'{self.n_labeled} old anchors + {self.n_unlabeled} novel slots'
            )
            centers, preds = self._seeded_fine_clustering(
                base_feats_np,
                all_labels.numpy(),
                all_masks.numpy().astype(bool),
            )
        else:
            print(f'  Level {deepest_layer}: KMeans with K={n_clusters}')
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            preds = kmeans.fit_predict(base_feats_np)
            centers = kmeans.cluster_centers_
        
        self.prototypes[deepest_layer] = torch.from_numpy(centers).float()
        self.pseudo_labels[deepest_layer] = torch.from_numpy(preds).long()
        self.fine_to_level[deepest_layer] = torch.arange(n_clusters, dtype=torch.long)
        self._compute_radii(deepest_layer, all_features[deepest_layer])
        
        # Step 3: Build coarser levels in SelEx style:
        # 1) Merge previous "known" prototypes into fewer known groups
        # 2) Treat samples assigned to previous known clusters as fixed-labeled
        # 3) Run constrained KMeans over the same base feature space
        prev_layer = deepest_layer
        prev_known_slots = self.n_labeled
        prev_total_slots = self.n_clusters_per_level[deepest_layer]
        for layer_idx in reversed(self.extract_layers[:-1]):
            n_clusters = self.n_clusters_per_level[layer_idx]
            known_slots = int(round(n_clusters * (prev_known_slots / max(prev_total_slots, 1))))
            known_slots = max(1, min(known_slots, n_clusters - 1))
            print(
                f'  Level {layer_idx}: SelEx-style constrained clustering '
                f'K={n_clusters} (known={known_slots}, novel={n_clusters - known_slots})'
            )

            # Merge previous known prototypes -> current known groups
            prev_protos_np = self.prototypes[prev_layer].numpy()
            prev_pseudo = self.pseudo_labels[prev_layer].numpy()
            prev_known_proto = prev_protos_np[:prev_known_slots]
            known_merger = KMeans(n_clusters=known_slots, random_state=0, n_init=10)
            known_merger.fit(prev_known_proto)
            merged_known_labels = known_merger.labels_

            prev_known_mask = prev_pseudo < prev_known_slots
            fixed_mask = prev_known_mask.copy()
            fixed_labels = np.zeros_like(prev_pseudo)
            fixed_labels[fixed_mask] = merged_known_labels[prev_pseudo[fixed_mask]]

            centers_np, refined_preds = self._semi_supervised_level_clustering(
                features=base_feats_np,
                fixed_mask=fixed_mask,
                fixed_labels=fixed_labels,
                n_clusters=n_clusters,
                n_known=known_slots,
            )

            self.prototypes[layer_idx] = torch.from_numpy(centers_np).float()
            self.pseudo_labels[layer_idx] = torch.from_numpy(refined_preds).long()
            self.fine_to_level[layer_idx] = self._majority_map_from_fine(
                fine_labels=self.pseudo_labels[deepest_layer].numpy(),
                level_labels=refined_preds,
                n_fine=self.n_clusters_per_level[deepest_layer],
                n_level=n_clusters,
            )
            self._compute_radii(layer_idx, all_features[layer_idx])

            prev_layer = layer_idx
            prev_known_slots = known_slots
            prev_total_slots = n_clusters
        
        model.train()
        print(f'  Hierarchy built: {self.get_hierarchy_summary()}')

    def _seeded_fine_clustering(self, features, labels, labelled_mask):
        n_clusters = self.n_clusters_per_level[max(self.extract_layers)]
        if n_clusters != self.n_labeled + self.n_unlabeled:
            raise ValueError(
                "Label-anchored hierarchy expects the finest level to have "
                f"{self.n_labeled + self.n_unlabeled} clusters, got {n_clusters}"
            )

        old_centers = []
        fallback_kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        fallback_kmeans.fit(features)
        fallback_centers = fallback_kmeans.cluster_centers_

        for class_idx in range(self.n_labeled):
            class_mask = labelled_mask & (labels == class_idx)
            if class_mask.any():
                old_centers.append(features[class_mask].mean(axis=0))
            else:
                old_centers.append(fallback_centers[class_idx])

        if self.n_unlabeled > 0:
            residual = features[~labelled_mask]
            if len(residual) < self.n_unlabeled:
                residual = features
            novel_kmeans = KMeans(n_clusters=self.n_unlabeled, random_state=0, n_init=10)
            novel_kmeans.fit(residual)
            centers = np.concatenate([np.stack(old_centers), novel_kmeans.cluster_centers_], axis=0)
        else:
            centers = np.stack(old_centers)

        centers = self._l2_normalize_np(centers)
        labelled_old_mask = labelled_mask & (labels < self.n_labeled)
        preds = np.zeros(len(features), dtype=np.int64)

        for _ in range(self.seeded_kmeans_iters):
            sims = features @ centers.T
            preds = sims.argmax(axis=1).astype(np.int64)
            preds[labelled_old_mask] = labels[labelled_old_mask].astype(np.int64)

            new_centers = np.zeros_like(centers)
            for cluster_idx in range(n_clusters):
                assigned = preds == cluster_idx
                if cluster_idx < self.n_labeled:
                    anchor_mask = labelled_mask & (labels == cluster_idx)
                    if anchor_mask.any():
                        anchor = features[anchor_mask].mean(axis=0, keepdims=True)
                        if assigned.any():
                            assigned_sum = features[assigned].sum(axis=0, keepdims=True)
                            numerator = assigned_sum + self.label_anchor_weight * anchor
                            denom = assigned.sum() + self.label_anchor_weight
                            new_centers[cluster_idx] = (numerator / denom)[0]
                        else:
                            new_centers[cluster_idx] = anchor[0]
                    elif assigned.any():
                        new_centers[cluster_idx] = features[assigned].mean(axis=0)
                    else:
                        new_centers[cluster_idx] = centers[cluster_idx]
                elif assigned.any():
                    new_centers[cluster_idx] = features[assigned].mean(axis=0)
                else:
                    new_centers[cluster_idx] = centers[cluster_idx]
            centers = self._l2_normalize_np(new_centers)

        sims = features @ centers.T
        preds = sims.argmax(axis=1).astype(np.int64)
        preds[labelled_old_mask] = labels[labelled_old_mask].astype(np.int64)
        return centers, preds

    def _semi_supervised_level_clustering(self, features, fixed_mask, fixed_labels, n_clusters, n_known):
        n_novel = n_clusters - n_known
        fixed_mask = fixed_mask.astype(bool)
        fixed_labels = fixed_labels.astype(np.int64)

        known_centers = []
        for cls_idx in range(n_known):
            cls_mask = fixed_mask & (fixed_labels == cls_idx)
            if cls_mask.any():
                known_centers.append(features[cls_mask].mean(axis=0))
            else:
                known_centers.append(features[np.random.randint(0, len(features))])
        known_centers = np.stack(known_centers, axis=0)

        residual = features[~fixed_mask]
        if len(residual) < max(n_novel, 1):
            residual = features
        if n_novel > 0:
            novel_kmeans = KMeans(n_clusters=n_novel, random_state=0, n_init=10)
            novel_kmeans.fit(residual)
            centers = np.concatenate([known_centers, novel_kmeans.cluster_centers_], axis=0)
        else:
            centers = known_centers
        centers = self._l2_normalize_np(centers)

        preds = np.zeros(len(features), dtype=np.int64)
        for _ in range(self.seeded_kmeans_iters):
            sims = features @ centers.T
            preds = sims.argmax(axis=1).astype(np.int64)
            preds[fixed_mask] = fixed_labels[fixed_mask]

            new_centers = np.zeros_like(centers)
            for cluster_idx in range(n_clusters):
                assigned = preds == cluster_idx
                if assigned.any():
                    new_centers[cluster_idx] = features[assigned].mean(axis=0)
                else:
                    new_centers[cluster_idx] = centers[cluster_idx]
            centers = self._l2_normalize_np(new_centers)

        sims = features @ centers.T
        preds = sims.argmax(axis=1).astype(np.int64)
        preds[fixed_mask] = fixed_labels[fixed_mask]
        return centers, preds

    @staticmethod
    def _l2_normalize_np(array):
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return array / norms

    @staticmethod
    def _majority_map_from_fine(fine_labels, level_labels, n_fine, n_level):
        mapping = np.zeros(n_fine, dtype=np.int64)
        for fine_idx in range(n_fine):
            mask = fine_labels == fine_idx
            if mask.any():
                counts = np.bincount(level_labels[mask], minlength=n_level)
                mapping[fine_idx] = int(counts.argmax())
        return torch.from_numpy(mapping).long()
    
    def _compute_radii(self, layer_idx, features):
        """Compute cluster radii for a given level."""
        pseudo = self.pseudo_labels[layer_idx]
        protos = self.prototypes[layer_idx]
        n_clusters = protos.shape[0]
        
        radii = torch.zeros(n_clusters)
        for c in range(n_clusters):
            mask = pseudo == c
            if mask.sum() > 0:
                dists = torch.cdist(
                    features[mask], protos[c:c+1]
                ).squeeze(1)
                radii[c] = dists.mean()
        
        self.cluster_radii[layer_idx] = radii
    
    def get_pseudo_labels(self, layer_idx, indices=None):
        """
        Get pseudo labels for a given level.
        
        Args:
            layer_idx: DINO block index
            indices: Optional sample indices. If None, return all.
        
        Returns:
            Pseudo labels tensor
        """
        labels = self.pseudo_labels[layer_idx]
        if indices is not None:
            positions = self._resolve_positions(indices)
            return labels[positions]
        return labels

    def get_fine_to_level_mapping(self, layer_idx, device=None):
        mapping = self.fine_to_level[layer_idx]
        if device is not None:
            mapping = mapping.to(device)
        return mapping

    def _resolve_positions(self, indices):
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        return torch.tensor(
            [self.index_to_position[int(sample_idx)] for sample_idx in indices],
            dtype=torch.long,
        )

    def get_confusion_weights(self, features_dict, sample_indices, device='cuda', n_views=2,
                              mode='multi'):
        """
        Compute semantic-aware confusion weights using hierarchy.
        
        For each pair of samples, compute a weight based on how many
        hierarchy levels they share the same cluster in:
        - Same cluster at all levels → high confusion (likely same class)
        - Different at coarse level → low confusion (clearly different)
        
        Args:
            features_dict: dict of layer_idx → features [B, dim]
            sample_indices: indices of current batch samples
            device: torch device
        
        Returns:
            confusion_weights: [n_views * B, n_views * B] matrix
        """
        B = len(sample_indices)
        confusion = torch.zeros(B, B, device=device)
        positions = self._resolve_positions(sample_indices)

        active_layers = [self.extract_layers[0]] if mode == 'coarse' else self.extract_layers
        if mode not in ('multi', 'coarse'):
            raise ValueError(f"Unknown relation relaxation mode: {mode}")

        num_levels = len(active_layers)
        for i, layer_idx in enumerate(active_layers):
            pseudo = self.pseudo_labels[layer_idx][positions].to(device)
            # Same cluster → 1, different → 0
            same_cluster = (pseudo.unsqueeze(0) == pseudo.unsqueeze(1)).float()
            # Weight by hierarchy level (deeper = more weight)
            weight = 1.0 / (2 ** (num_levels - i - 1))
            confusion += weight * same_cluster
        
        # Normalize to [0, 1]
        confusion = confusion / confusion.max().clamp_min(1e-12)
        
        # Expand for multi-view batches: [B, B] → [n_views * B, n_views * B]
        confusion = confusion.repeat(n_views, n_views)
        
        return confusion
    
    def get_hierarchy_summary(self):
        """Return a readable summary of the hierarchy."""
        parts = []
        for layer_idx in sorted(self.extract_layers):
            n = self.n_clusters_per_level[layer_idx]
            parts.append(f'L{layer_idx}={n}')
        return ' → '.join(parts)
    
    def is_built(self):
        """Check if hierarchy has been built."""
        return len(self.pseudo_labels) > 0
