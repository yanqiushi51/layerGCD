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
    
    def __init__(self, extract_layers, n_labeled, n_unlabeled, min_classes=8):
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
        
        # Step 2: Bottom-up hierarchy construction
        # Start from the deepest DINO layer (finest clustering)
        deepest_layer = max(self.extract_layers)
        feats_np = all_features[deepest_layer].numpy()
        
        n_clusters = self.n_clusters_per_level[deepest_layer]
        print(f'  Level {deepest_layer}: KMeans with K={n_clusters}')
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        preds = kmeans.fit_predict(feats_np)
        
        self.prototypes[deepest_layer] = torch.from_numpy(kmeans.cluster_centers_).float()
        self.pseudo_labels[deepest_layer] = torch.from_numpy(preds).long()
        self._compute_radii(deepest_layer, all_features[deepest_layer])
        
        # Step 3: Build coarser levels by clustering prototypes
        prev_layer = deepest_layer
        for layer_idx in reversed(self.extract_layers[:-1]):
            n_clusters = self.n_clusters_per_level[layer_idx]
            print(f'  Level {layer_idx}: KMeans with K={n_clusters}')
            
            # Use this layer's features for clustering
            feats_np = all_features[layer_idx].numpy()
            
            # Cluster the previous level's prototypes to get mapping
            prev_protos_np = self.prototypes[prev_layer].numpy()
            proto_kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            proto_mapping = proto_kmeans.fit_predict(prev_protos_np)
            
            # Map sample labels: previous pseudo-label → coarse init labels
            prev_pseudo = self.pseudo_labels[prev_layer].numpy()
            coarse_init_labels = np.array([proto_mapping[p] for p in prev_pseudo])

            # Initialize coarse prototypes using THIS layer's features, then
            # refine assignments on this layer so pseudo-labels are not just
            # inherited from the previous level's prototype graph.
            new_protos = np.zeros((n_clusters, feats_np.shape[1]), dtype=feats_np.dtype)
            for c in range(n_clusters):
                mask_c = coarse_init_labels == c
                if mask_c.sum() > 0:
                    new_protos[c] = feats_np[mask_c].mean(axis=0)
            
            # Normalize prototypes
            norms = np.linalg.norm(new_protos, axis=1, keepdims=True)
            norms[norms == 0] = 1
            new_protos = new_protos / norms

            layer_kmeans = KMeans(
                n_clusters=n_clusters,
                init=new_protos,
                n_init=1,
                random_state=0,
            )
            refined_preds = layer_kmeans.fit_predict(feats_np)

            self.prototypes[layer_idx] = torch.from_numpy(layer_kmeans.cluster_centers_).float()
            self.pseudo_labels[layer_idx] = torch.from_numpy(refined_preds).long()
            self._compute_radii(layer_idx, all_features[layer_idx])
            
            prev_layer = layer_idx
        
        model.train()
        print(f'  Hierarchy built: {self.get_hierarchy_summary()}')
    
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
