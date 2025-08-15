
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

class AttentionVisualizer:
    def __init__(self):
        self.collected_attentions = []
        self.hook_handles = []

    def _attention_hook_fn(self, module, input_args, output_val):
        if hasattr(module, 'attn_probs_softmax') and module.attn_probs_softmax is not None:
            self.collected_attentions.append(module.attn_probs_softmax.cpu())
        else:
            print(f"Warning: Module {type(module).__name__} does not have the 'attn_probs_softmax' attribute or its value is None.")

    def register_hooks_on_attention_modules(self, model_to_visualize, s_a_block_lists):
        self.clear_hooks()
        if not isinstance(s_a_block_lists, list):
            s_a_block_lists = [s_a_block_lists]

        for block_list_container in s_a_block_lists:
            if block_list_container is None: continue
            for block_module in block_list_container:

                if hasattr(block_module, 'attn') and isinstance(block_module.attn, torch.nn.Module):
                    attention_submodule = block_module.attn
                    handle = attention_submodule.register_forward_hook(self._attention_hook_fn)
                    self.hook_handles.append(handle)

        if not self.hook_handles:
            print("Warning: Failed to register any hooks to the Attention module. Please check s_a_block_lists and the model structure.")

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.collected_attentions = []

    def get_attentions_for_forward_pass(self,
                                         model_to_run_forward_on,
                                         model_containing_actual_blocks,
                                         list_of_modulelists_to_hook,
                                         *model_fwd_args, **model_fwd_kwargs):

        self.register_hooks_on_attention_modules(
            model_containing_actual_blocks,
            list_of_modulelists_to_hook
        )

        if not self.hook_handles:
            print("Unable to perform forward propagation to collect attention due to unsuccessful hook registration.")
            self.clear_hooks()
            return []

        self.collected_attentions = []
        if isinstance(model_to_run_forward_on, torch.nn.Module):
            model_to_run_forward_on.eval()
        with torch.no_grad():
            _ = model_to_run_forward_on(*model_fwd_args, **model_fwd_kwargs)
        attentions_to_return = list(self.collected_attentions)
        self.clear_hooks()
        return attentions_to_return

    def compute_rollout(self, attentions_for_one_sample_all_layers, head_fusion="mean", include_residual=True):
        if not attentions_for_one_sample_all_layers:
            return None

        num_tokens = attentions_for_one_sample_all_layers[0].shape[-1]
        rollout_attention = torch.eye(num_tokens, dtype=torch.float32)
        for attn_matrix_heads in attentions_for_one_sample_all_layers:

            if head_fusion == "mean":
                attn_matrix_layer = torch.mean(attn_matrix_heads, dim=0)
            elif head_fusion == "max":
                attn_matrix_layer = torch.max(attn_matrix_heads, dim=0)[0]
            else:
                attn_matrix_layer = torch.mean(attn_matrix_heads, dim=0)

            if include_residual:
                identity = torch.eye(num_tokens, dtype=torch.float32)
                attn_matrix_layer = 0.5 * attn_matrix_layer + 0.5 * identity
            rollout_attention = torch.matmul(attn_matrix_layer, rollout_attention)
        return rollout_attention

    def visualize_2d_attention(self, attention_scores_1d,
                                patch_grid_shape,
                                original_image_slice_for_overlay,
                                save_path="attention_viz.png",
                                alpha=0.6,
                                save_original=True,
                                save_heatmap_only=True):

        H_feat, W_feat = patch_grid_shape
        num_patches_expected = H_feat * W_feat

        if not isinstance(attention_scores_1d, (np.ndarray, torch.Tensor)):
            print(f"Error: attention_scores_1d must be a NumPy array or a PyTorch tensor, but got {type(attention_scores_1d)}")
            return
        if attention_scores_1d.size == 0:
            print(f"Error: attention_scores_1d is empty.")
            return
        if attention_scores_1d.shape[0] != num_patches_expected:
            print(f"Error: The length of attention scores ({attention_scores_1d.shape[0]}) does not match the expected number of patches {num_patches_expected} ({H_feat}x{W_feat}).")
            return

        if isinstance(attention_scores_1d, torch.Tensor):
            attention_scores_1d = attention_scores_1d.cpu().numpy()

        attention_map_2d = attention_scores_1d.reshape(H_feat, W_feat)

        map_min, map_max = attention_map_2d.min(), attention_map_2d.max()
        if map_max > map_min:
            attention_map_2d_norm = (attention_map_2d - map_min) / (map_max - map_min)
        else:
            attention_map_2d_norm = np.zeros_like(attention_map_2d) if map_min == 0 else np.ones_like(attention_map_2d) * map_min

        h_orig, w_orig = original_image_slice_for_overlay.shape[:2]
        heatmap_resized_norm = cv2.resize(attention_map_2d_norm, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        heatmap_colored_bgr = cv2.applyColorMap((heatmap_resized_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        if original_image_slice_for_overlay.ndim == 2:

            img_for_overlay_rgb_uint8 = (np.stack([original_image_slice_for_overlay]*3, axis=-1) * 255).astype(np.uint8)
        elif original_image_slice_for_overlay.ndim == 3 and original_image_slice_for_overlay.shape[-1] == 1:
            img_for_overlay_rgb_uint8 = (np.concatenate([original_image_slice_for_overlay]*3, axis=-1) * 255).astype(np.uint8)
        elif original_image_slice_for_overlay.ndim == 3 and original_image_slice_for_overlay.shape[-1] == 3:
            img_for_overlay_rgb_uint8 = (original_image_slice_for_overlay * 255).astype(np.uint8)
        else:
            print(f"Error: The dimensions of original_image_slice_for_overlay {original_image_slice_for_overlay.shape} are not supported. Expected (H, W) or (H, W, 3) with values in the range 0-1.")
            return

        blended_img_rgb = cv2.addWeighted(img_for_overlay_rgb_uint8, 1 - alpha, heatmap_colored_rgb, alpha, 0)

        base, ext = os.path.splitext(save_path)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            print(f"Warning: The extension of save_path '{save_path}' is unknown, defaulting to .png")
            ext = '.png'
            save_path = base + ext

        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        plt.figure()
        plt.imshow(blended_img_rgb)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved [overlayed] visualized attention map to: {save_path}")

        if save_original:
            original_save_path = f"{base}_original{ext}"
            plt.figure()
            plt.imshow(img_for_overlay_rgb_uint8)
            plt.axis('off')
            plt.savefig(original_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        if save_heatmap_only:
            heatmap_only_save_path = f"{base}_heatmap_only{ext}"
            plt.figure()
            plt.imshow(heatmap_colored_rgb)
            plt.axis('off')
            plt.savefig(heatmap_only_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()


    class VizModelWrapper(torch.nn.Module):
        def __init__(self, actual_model_instance, fixed_secondary_input):
            super().__init__()
            self.actual_model = actual_model_instance
            self.fixed_secondary_input = fixed_secondary_input

        def forward(self, primary_image_input):

            return self.actual_model(primary_image_input, self.fixed_secondary_input)