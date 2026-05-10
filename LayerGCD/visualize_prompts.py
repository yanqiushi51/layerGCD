import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import PromptGuidedDINO


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def prepare_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img = Image.open(image_path).convert('RGB')
    orig_img = np.array(img.resize((image_size, image_size)))
    
    tensor_img = transform(img).unsqueeze(0)
    return tensor_img, orig_img

def plot_attention(img, attention, save_path, title="Attention"):
    # attention shape: [14, 14]
    attention = attention - attention.min()
    if attention.max() > 0:
        attention = attention / attention.max()
        
    attention = cv2.resize(attention, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cam)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def infer_model_config(state_dict):
    model_state = state_dict.get('model', state_dict)
    config = state_dict.get('config', {})
    num_prompt_tokens = model_state['P_coarse'].shape[1]
    num_coarse_classes = model_state['coarse_head.last_layer.weight_v'].shape[0]
    num_classes = model_state['fine_head.last_layer.weight_v'].shape[0]
    extract_layers = sorted(set(config.get('extract_layers', [7, 9, 11])))
    if 12 in extract_layers:
        extract_layers.remove(12)
        extract_layers.append(11)
    extract_layers = sorted(set(extract_layers))
    return num_classes, num_coarse_classes, num_prompt_tokens, extract_layers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=200, help='Total classes')
    parser.add_argument('--num_coarse_classes', type=int, default=None, help='Coarse prompt head classes')
    parser.add_argument('--num_prompt_tokens', type=int, default=4, help='Prompt tokens per branch')
    parser.add_argument('--extract_layers', nargs='+', type=int, default=[7, 9, 11], help='Prompt checkpoint blocks')
    parser.add_argument('--out_dir', type=str, default='./vis_results', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_state = None
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
        ckpt_num_classes, ckpt_num_coarse_classes, ckpt_num_prompt_tokens, ckpt_extract_layers = infer_model_config(checkpoint_state)
        if args.num_coarse_classes is None:
            args.num_coarse_classes = ckpt_num_coarse_classes
        args.num_classes = ckpt_num_classes
        args.num_prompt_tokens = ckpt_num_prompt_tokens
        args.extract_layers = ckpt_extract_layers

    # Load Model
    print("Loading PromptGuidedDINO...")
    model = PromptGuidedDINO(
        num_classes=args.num_classes,
        extract_layers=args.extract_layers,
        num_coarse_classes=args.num_coarse_classes,
        num_prompt_tokens=args.num_prompt_tokens,
    ).to(device)
    
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state.get('model', checkpoint_state))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No valid checkpoint found, running with random prompt initialization.")

    model.eval()

    # Load Image
    tensor_img, orig_img = prepare_image(args.image_path)
    tensor_img = tensor_img.to(device)

    # We need to hook the attention maps. In DINO ViT, the attention is computed in `attn`.
    # `attn(x)` returns x. We need the attention weights before projection.
    # DINO ViT `Attention` module usually has a specific structure.
    # We will hook into the `qkv` projection to manually compute attention, or try to hook `attn_drop`.
    
    # The prompt checkpoint blocks come from the training config/checkpoint metadata.
    coarse_idx = model.coarse_layer_idx
    fine_idx = model.fine_layer_idx
    
    # Let's write a custom hook to grab the attention maps directly
    def get_attention(block, x):
        B, N, C = x.shape
        # qkv(): [B, N, 3*C] -> reshape -> [3, B, num_heads, N, head_dim]
        qkv = block.attn.qkv(x).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn = attn.softmax(dim=-1)
        # attn shape: [B, num_heads, N, N]
        # Average across heads: [B, N, N]
        return attn.mean(dim=1)

    print("Running inference...")
    with torch.no_grad():
        x = model.dino.prepare_tokens(tensor_img)
        B = x.shape[0]
        N_p = model.num_prompt_tokens
        
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        p_c = model.P_coarse.expand(B, -1, -1)
        p_f = model.P_fine.expand(B, -1, -1)
        
        # Sequence: [CLS, P_c_1..P_c_N, P_f_1..P_f_N, patches]
        x = torch.cat([cls_token, p_c, p_f, patch_tokens], dim=1)
        
        for i, blk in enumerate(model.dino.blocks):
            attn_matrix = get_attention(blk, x)[0]
            x = blk(x)

            if i == coarse_idx:
                coarse_raw = x[:, 1:1+N_p, :].mean(dim=1)
                coarse_feat = model.dino.norm(coarse_raw)
                prior_msg = model.bridge_mlp(coarse_feat)

                fine_start = 1 + N_p
                fine_end = 1 + 2 * N_p
                x_p_fine_new = x[:, fine_start:fine_end, :] + prior_msg.unsqueeze(1)
                x = torch.cat([x[:, :fine_start, :], x_p_fine_new, x[:, fine_end:, :]], dim=1)

                # Attention of the coarse checkpoint block.
                macro_attn_matrix = attn_matrix

            if i == fine_idx:
                # Attention of the final fine block.
                micro_attn_matrix = attn_matrix

    # Sequence Layout: [CLS, P_coarse_1..N, P_fine_1..N, patches]
    patch_start = 1 + 2 * N_p
    p_coarse_to_patches = macro_attn_matrix[1:1+N_p, patch_start:].mean(dim=0).cpu().numpy()
    p_coarse_map = p_coarse_to_patches.reshape(14, 14)

    # Average attention from all fine prompt tokens to image patches.
    fine_start = 1 + N_p
    fine_end = 1 + 2 * N_p
    p_fine_to_patches = micro_attn_matrix[fine_start:fine_end, patch_start:].mean(dim=0).cpu().numpy()
    p_fine_map = p_fine_to_patches.reshape(14, 14)

    # Plot
    plot_attention(
        orig_img,
        p_coarse_map,
        os.path.join(args.out_dir, "macro_P_coarse_attention.png"),
        f"Block {coarse_idx}: Macro Attention (P_coarse)",
    )
    plot_attention(
        orig_img,
        p_fine_map,
        os.path.join(args.out_dir, "micro_P_fine_attention.png"),
        f"Block {fine_idx}: Micro Attention (P_fine)",
    )

if __name__ == '__main__':
    main()
