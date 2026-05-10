import torch
import torch.nn as nn
import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class MultiLayerDINO(nn.Module):
    """
    Wrapper around DINO ViT-B/16 that exposes intermediate layer CLS tokens.
    
    Instead of only returning the final CLS token, this module hooks into
    specified intermediate blocks and returns CLS tokens at each level.
    This enables building a natural semantic hierarchy:
      - Lower layers → fine-grained features (textures, parts)
      - Higher layers → abstract features (category-level semantics)
    """
    
    def __init__(self, backbone, extract_layers=(7, 9, 11)):
        """
        Args:
            backbone: A DINO ViT model (e.g., dino_vitb16)
            extract_layers: Tuple of 0-indexed block indices to extract CLS from.
                           The final block is always included automatically.
        """
        super().__init__()
        self.backbone = backbone
        self.extract_layers = sorted(extract_layers)
        
        # Determine total depth
        self.depth = len(backbone.blocks)
        
        # Final layer is always included
        if (self.depth - 1) not in self.extract_layers:
            self.extract_layers.append(self.depth - 1)
        
        self.feat_dim = backbone.embed_dim  # 768 for ViT-B
        
    def forward(self, x, return_all_layers=False):
        """
        Forward pass with multi-layer feature extraction.
        
        Args:
            x: Input images [B, 3, H, W]
            return_all_layers: If True, return features from all extract_layers.
                             If False, return only the final CLS token (SimGCD-compatible).
        
        Returns:
            If return_all_layers=True:
                dict mapping layer_idx → CLS token [B, feat_dim]
            If return_all_layers=False:
                Final CLS token [B, feat_dim]
        """
        # Prepare tokens (patch embed + CLS + pos encoding)
        x = self.backbone.prepare_tokens(x)
        
        layer_features = {}
        
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if return_all_layers and i in self.extract_layers:
                # Extract CLS token (index 0) after layer norm
                cls_token = self.backbone.norm(x)[:, 0]
                layer_features[i] = cls_token
        
        # Final output (always apply norm)
        x = self.backbone.norm(x)
        final_cls = x[:, 0]
        
        if return_all_layers:
            # Make sure final layer is in the dict
            layer_features[self.depth - 1] = final_cls
            return layer_features
        else:
            return final_cls
    
    def get_layer_keys(self):
        """Return sorted list of layer indices that produce features."""
        return sorted(self.extract_layers)
    
    def get_num_levels(self):
        """Return number of hierarchy levels."""
        return len(self.extract_layers)


def build_multilayer_dino(pretrained=True, extract_layers=(7, 9, 11), 
                          grad_from_block=7):
    """
    Build a MultiLayerDINO model.
    
    Args:
        pretrained: Whether to load pretrained DINO weights
        extract_layers: Which intermediate blocks to extract CLS from
        grad_from_block: Finetune only blocks >= this index
    
    Returns:
        MultiLayerDINO model
    """
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', 
                               pretrained=pretrained)
    
    # Freeze early layers
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Unfreeze from grad_from_block onwards
    for name, param in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_from_block:
                param.requires_grad = True
    
    model = MultiLayerDINO(backbone, extract_layers=list(extract_layers))
    return model
