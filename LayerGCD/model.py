import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    """Projection + classification head (from SimGCD)."""
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return x_proj, logits


class HierarchicalHead(nn.Module):
    """
    Multi-level projection + classification heads.
    
    Each hierarchy level (corresponding to a DINO layer) gets its own
    DINOHead with an appropriate number of output classes.
    
    Level 0 (deepest DINO layer, e.g. Layer 12): full K classes
    Level 1 (Layer 11): K/2 classes  
    Level 2 (Layer 9):  K/4 classes
    Level 3 (Layer 7):  K/8 classes (coarsest, but finest DINO layer)
    
    Note: The levels are ordered from DINO's perspective:
    - extract_layers[0] (e.g., 7) = coarsest hierarchy = level with fewest classes
    - extract_layers[-1] (e.g., 12) = finest hierarchy = full K classes
    """
    
    def __init__(self, feat_dim, num_classes, extract_layers, 
                 nlayers=3, min_classes=8):
        """
        Args:
            feat_dim: Feature dimension (768 for ViT-B)
            num_classes: Total number of classes (K = labeled + unlabeled)
            extract_layers: Sorted list of DINO block indices
            nlayers: Number of MLP layers in each head
            min_classes: Minimum number of classes at coarsest level
        """
        super().__init__()
        
        self.extract_layers = extract_layers
        self.num_levels = len(extract_layers)
        self.feat_dim = feat_dim
        
        # Compute number of classes per level
        # Deepest DINO layer → full K classes
        # Each level above → halve the classes
        self.num_classes_per_level = {}
        for i, layer_idx in enumerate(reversed(extract_layers)):
            # i=0 is the deepest layer (full classes)
            n_cls = max(num_classes // (2 ** i), min_classes)
            self.num_classes_per_level[layer_idx] = n_cls
        
        # Create a DINOHead for each level
        self.heads = nn.ModuleDict()
        for layer_idx in extract_layers:
            n_cls = self.num_classes_per_level[layer_idx]
            self.heads[str(layer_idx)] = DINOHead(
                in_dim=feat_dim,
                out_dim=n_cls,
                nlayers=nlayers
            )
    
    def forward(self, layer_features):
        """
        Args:
            layer_features: dict mapping layer_idx → CLS token [B, feat_dim]
        
        Returns:
            projections: dict mapping layer_idx → projected features [B, bottleneck_dim]
            logits: dict mapping layer_idx → classification logits [B, n_classes_at_level]
        """
        projections = {}
        logits = {}
        
        for layer_idx, feat in layer_features.items():
            key = str(layer_idx)
            if key in self.heads:
                proj, logit = self.heads[key](feat)
                projections[layer_idx] = proj
                logits[layer_idx] = logit
        
        return projections, logits
    
    def get_global_head(self):
        """Return the head for the deepest (global) level."""
        global_layer = max(self.extract_layers)
        return self.heads[str(global_layer)]
    
    def get_num_classes(self, layer_idx):
        """Return number of classes for a given layer."""
        return self.num_classes_per_level[layer_idx]


class PromptGuidedDINO(nn.Module):
    """
    Prompt-Guided Hierarchical Vision Transformer.
    Freezes the DINO backbone and uses two learnable prompts (P_coarse, P_fine) 
    that navigate the layers. P_coarse extracts macro features and injects a prior 
    into P_fine.
    """
    def __init__(self, num_classes, extract_layers=(7, 11), num_coarse_classes=None,
                 num_prompt_tokens=4, disable_bridge=False, fine_prompt_only=False,
                 no_prompts=False):
        super().__init__()
        from network import build_multilayer_dino

        extract_layers = sorted(set(extract_layers))
        if 12 in extract_layers:
            extract_layers.remove(12)
            extract_layers.append(11)
        extract_layers = sorted(set(extract_layers))
        
        # Load pre-trained DINO via our wrapper to easily extract features for the hierarchy tree
        # We set grad_from_block=12 to freeze everything completely.
        self.dino_feature_extractor = build_multilayer_dino(
            pretrained=True, 
            extract_layers=extract_layers,
            grad_from_block=11  # Fix 3: unfreeze last block
        )
        self.dino = self.dino_feature_extractor.backbone
        
        self.num_prompt_tokens = num_prompt_tokens
        self.disable_bridge = disable_bridge
        self.fine_prompt_only = fine_prompt_only
        self.no_prompts = no_prompts
            
        # Multi-token learnable prompts for stronger representational capacity
        if no_prompts:
            self.register_parameter('P_coarse', None)
            self.register_parameter('P_fine', None)
        elif not fine_prompt_only:
            self.P_coarse = nn.Parameter(torch.randn(1, num_prompt_tokens, 768))
            nn.init.normal_(self.P_coarse, std=0.02)
            self.P_fine = nn.Parameter(torch.randn(1, num_prompt_tokens, 768))
            nn.init.normal_(self.P_fine, std=0.02)
        else:
            self.register_parameter('P_coarse', None)
            self.P_fine = nn.Parameter(torch.randn(1, num_prompt_tokens, 768))
            nn.init.normal_(self.P_fine, std=0.02)
        
        self.coarse_layer_idx = extract_layers[0]
        self.fine_layer_idx = extract_layers[-1]
        
        if num_coarse_classes is None:
            num_coarse_classes = num_classes

        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_classes

        # Two distinct heads for tracking macro and micro concepts
        coarse_head_dim = 768 * 2
        self.coarse_head = None if (fine_prompt_only or no_prompts) else DINOHead(
            in_dim=coarse_head_dim, out_dim=num_coarse_classes, nlayers=3
        )
        fine_head_dim = 768 if no_prompts else 768 * 2
        self.fine_head = DINOHead(in_dim=fine_head_dim, out_dim=num_classes, nlayers=3)
        
        # Bridge MLP to pass structural priors from P_coarse to P_fine
        self.bridge_mlp = nn.Sequential(
            nn.Linear(coarse_head_dim, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        # Fix 2: zero-init last layer so bridge starts as identity
        nn.init.constant_(self.bridge_mlp[-1].weight, 0)
        nn.init.constant_(self.bridge_mlp[-1].bias, 0)
        
    def forward(self, x):
        B = x.shape[0]
        N_p = self.num_prompt_tokens
        
        # Convert images to patch tokens + CLS token
        x = self.dino.prepare_tokens(x)

        if self.no_prompts:
            for blk in self.dino.blocks:
                x = blk(x)
            x = self.dino.norm(x)
            fine_feat = x[:, 0, :]
            fine_proj, fine_logits = self.fine_head(fine_feat)
            return None, fine_logits, None, fine_feat, None, fine_proj
        
        # Split tokens to insert Prompts
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        p_f = self.P_fine.expand(B, -1, -1)
        
        if self.fine_prompt_only:
            # Sequence: [CLS, P_f_1..P_f_N, Patch_1..Patch_196]
            x = torch.cat([cls_token, p_f, patch_tokens], dim=1)
            fine_start = 1
            fine_end = 1 + N_p
        else:
            p_c = self.P_coarse.expand(B, -1, -1)
            # Sequence: [CLS, P_c_1..P_c_N, P_f_1..P_f_N, Patch_1..Patch_196]
            x = torch.cat([cls_token, p_c, p_f, patch_tokens], dim=1)
            fine_start = 1 + N_p
            fine_end = 1 + 2 * N_p
        
        coarse_feat = None
        coarse_logits = None
        coarse_proj = None
        
        # Iterate through frozen transformer blocks. Coarse prompt is read after
        # the designated block has been applied, then its prior is injected into
        # the fine prompt for the remaining depth.
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)

            if (not self.fine_prompt_only) and i == self.coarse_layer_idx:
                # Extract P_coarse tokens after the coarse checkpoint block.
                # Average-pool the N_p coarse tokens into a single representation.
                coarse_prompt = x[:, 1:1+N_p, :].mean(dim=1)
                coarse_prompt = self.dino.norm(coarse_prompt)
                coarse_cls = self.dino.norm(x)[:, 0, :]
                coarse_feat = torch.cat([coarse_cls, coarse_prompt], dim=-1)

                # Get coarse supervised classification / projection
                coarse_proj, coarse_logits = self.coarse_head(coarse_feat)

                # Information Injection: MLP(P_coarse) -> each P_fine token
                prior_msg = None if self.disable_bridge else self.bridge_mlp(coarse_feat)  # [B, 768]

                # Avoid inplace operation which breaks PyTorch autograd
                if prior_msg is not None:
                    x_p_fine_new = x[:, fine_start:fine_end, :] + prior_msg.unsqueeze(1)
                    x = torch.cat([x[:, :fine_start, :], x_p_fine_new, x[:, fine_end:, :]], dim=1)
            
        # Final layer normalization
        x = self.dino.norm(x)
        
        # Fix 1: concat DINO [CLS] with P_fine for maximum power
        cls_feat = x[:, 0, :]
        fine_prompt_feat = x[:, fine_start:fine_end, :].mean(dim=1)
        fine_feat = torch.cat([cls_feat, fine_prompt_feat], dim=-1)  # [B, 1536]
        fine_proj, fine_logits = self.fine_head(fine_feat)
        
        return coarse_logits, fine_logits, coarse_feat, fine_feat, coarse_proj, fine_proj


# ============================================================
# Loss Functions
# ============================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning (from SimGCD/SupContrast)."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device if features.is_cuda else torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda', confusion_factor=None):
    """Compute InfoNCE logits for unsupervised contrastive learning."""
    if features.size(0) % n_views != 0:
        raise ValueError('features.size(0) must be divisible by n_views')

    b_ = int(features.size(0) // n_views)
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    if confusion_factor is not None:
        confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)
        # Added to similarity_matrix to reduce repulsion penalty (Semantic Aware Repulsion)
        similarity_matrix = similarity_matrix + 0.5 * confusion_factor

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    positive_logits = torch.logsumexp(positives / temperature, dim=1, keepdim=True)
    negative_logits = negatives / temperature

    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, labels


class DistillLoss(nn.Module):
    """Self-distillation loss (from SimGCD)."""
    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ncrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for _ in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


def get_params_groups(backbone_model, head_model, base_lr=0.1, backbone_lr_scale=0.1):
    """
    Get parameter groups for optimizer.
    Backbone parameters get a lower learning rate.
    """
    backbone_reg = []
    backbone_not_reg = []
    head_reg = []
    head_not_reg = []
    
    for name, param in backbone_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            backbone_not_reg.append(param)
        else:
            backbone_reg.append(param)
    
    for name, param in head_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            head_not_reg.append(param)
        else:
            head_reg.append(param)
    
    return [
        {'params': head_reg, 'lr': base_lr},
        {'params': head_not_reg, 'lr': base_lr, 'weight_decay': 0.},
        {'params': backbone_reg, 'lr': base_lr * backbone_lr_scale},
        {'params': backbone_not_reg, 'lr': base_lr * backbone_lr_scale, 'weight_decay': 0.}
    ]
