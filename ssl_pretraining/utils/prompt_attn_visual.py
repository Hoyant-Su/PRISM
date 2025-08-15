
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import imageio
import glob

def visualize_and_save_attention(attn_weights, save_dir, input_shape, patch_shape, original_image, prefix='attn', batch_idx=0, step=-1, prompt="", case_id = ""):
    B, Q_len, K_len = attn_weights.shape
    D, H, W = input_shape
    D_out, H_out, W_out = patch_shape

    assert D_out * H_out * W_out == K_len, f"Mismatch: D_out*H_out*W_out={D_out*H_out*W_out}, but K_len={K_len}"

    attn_map = attn_weights[batch_idx].mean(0).reshape(1, 1, D_out, H_out, W_out)

    attn_map_upsampled = F.interpolate(attn_map, size=(D, H, W), mode='trilinear', align_corners=False)
    attn_map_upsampled = attn_map_upsampled.squeeze().cpu().detach().numpy()

    img = original_image[batch_idx]
    if img.dim() == 4:
        D, T, H, W = img.shape
        img = img[D//2]

    img = img.cpu().detach().numpy()

    os.makedirs(save_dir, exist_ok=True)

    for d in range(D):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img[d], cmap='gray')
        ax.imshow(attn_map_upsampled[d], cmap='jet', alpha=0.5)
        ax.axis('off')
        os.makedirs(f"{save_dir}/heatmap/{prompt}/images", exist_ok=True)
        save_path = os.path.join(save_dir, "heatmap", prompt, "images", f"{case_id}_{prefix}_step_{step}_b{batch_idx}_slice{d}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    video_dir = os.path.join(save_dir, "heatmap", prompt, "video")
    os.makedirs(video_dir, exist_ok=True)

    pattern = os.path.join(save_dir, "heatmap", prompt, "images", f"{case_id}_{prefix}_step_{step}_b{batch_idx}_slice*.png")
    image_paths = sorted(glob.glob(pattern), key=lambda x: int(x.split("slice")[-1].split(".")[0]))

    video_path = os.path.join(video_dir, f"{case_id}_{prefix}_step_{step}_b{batch_idx}.mp4")

    frames = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(video_path, frames, fps=5)