#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backbones.vit.vit_debug import VisionTransformer

plt.rcParams.update(
    {
        "font.size": 20,
        "font.weight": "bold",
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

class AttentionVisualizer:
    def __init__(self, model_path, model_name, gpu_id=0, backbone='vits'):
        """
        Initialize attention visualizer for ATTN-FIQA.

        Args:
            model_path: Path to pretrained model weights
            model_name: Name of the model file
            gpu_id: GPU device ID
            backbone: Model size ('vits' or 'vitb')
        """
        self.gpu_id = gpu_id
        self.backbone = backbone
        self.device = f"cuda:{gpu_id}"

        # Load model
        self.model = self._load_model(model_path, model_name, backbone)

        # Get model configuration
        if backbone == 'vits':
            self.num_blocks = 12
            self.num_heads = 8
            self.patch_size = 8
        elif backbone == 'vitb':
            self.num_blocks = 24
            self.num_heads = 16
            self.patch_size = 8
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.img_size = 112

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.num_patches_per_side = self.img_size // self.patch_size

    def _load_model(self, model_path, model_name, backbone):
        """Load pretrained model"""
        if backbone == 'vits':
            model = VisionTransformer(
                img_size=112,
                patch_size=8,
                num_classes=512,
                embed_dim=512,
                depth=12,
                mlp_ratio=5,
                num_heads=8,
                drop_path_rate=0.1,
                norm_layer="ln",
                mask_ratio=0.0
            )
            model.eval()
        elif backbone == 'vitb':
            model = VisionTransformer(
                img_size=112,
                patch_size=8,
                num_classes=512,
                embed_dim=512,
                depth=24,
                mlp_ratio=3,
                num_heads=16,
                drop_path_rate=0.1,
                norm_layer="ln",
                mask_ratio=0.0
            )
            model.eval()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        checkpoint_path = os.path.join(model_path, model_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        dict_checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'state_dict' in dict_checkpoint:
            state_dict = dict_checkpoint['state_dict']
        elif 'model' in dict_checkpoint:
            state_dict = dict_checkpoint['model']
        else:
            state_dict = dict_checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            elif k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)

        return model

    def load_and_preprocess_image(self, image_path, color_channel='RGB'):
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Resize to 112x112
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Convert color space if needed
        if color_channel == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save original for visualization
        original_image = image.copy()

        # Prepare for model input
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)

        # Normalize
        image_tensor = image_tensor.div(255.0).sub(0.5).div(0.5)

        return image_tensor, original_image

    def extract_attention_and_score(self, image_tensor):
        """
        Extract attention maps and quality score from image.

        Returns:
            score: Quality score (float)
            last_layer_attention: Attention from last layer, all heads concatenated
                                 Shape: (num_patches, num_patches)
        """
        with torch.no_grad():
            # Get scores and all attention values per block
            # Don't specify output_block_index to get all blocks
            scores, all_attention_per_block = self.model.calculate_attnfiqa(
                image_tensor,
            )

            # Check if we got attention values
            if not all_attention_per_block or len(all_attention_per_block) == 0:
                raise ValueError("No attention values returned from model")

            # Get last block attention
            # all_attention_per_block is list of tensors
            # Expected: (batch_size, num_heads, num_patches, num_patches)
            # But might be flattened: (batch_size, num_heads * num_patches * num_patches)
            last_block_attn = all_attention_per_block[-1]  # Last block

            # Check and reshape if needed
            if last_block_attn.dim() == 2:
                # Flattened format: (batch_size, num_heads * num_patches * num_patches)
                batch_size = last_block_attn.shape[0]
                expected_size = self.num_heads * self.num_patches * self.num_patches
                if last_block_attn.shape[1] == expected_size:
                    # Reshape to (batch_size, num_heads, num_patches, num_patches)
                    last_block_attn = last_block_attn.reshape(
                        batch_size, self.num_heads, self.num_patches, self.num_patches
                    )
                else:
                    raise ValueError(f"Cannot reshape attention: shape {last_block_attn.shape}, expected size {expected_size}")
            elif last_block_attn.dim() < 3:
                raise ValueError(f"Unexpected attention shape: {last_block_attn.shape}")

            # Remove batch dimension (we only have 1 image)
            if last_block_attn.dim() == 4:
                last_block_attn = last_block_attn[0]  # (num_heads, num_patches, num_patches)

            # Average across heads for visualization (all_concat means we use all heads)
            # Shape: (num_patches, num_patches)
            avg_attn = last_block_attn.mean(dim=0)

            score = scores[0].cpu().item()
            avg_attn = avg_attn.cpu().numpy()

        return score, avg_attn

    def create_attention_heatmap(self, attention_map, vmin=None, vmax=None):
        """
        Convert attention map to heatmap overlay.

        Args:
            attention_map: (num_patches, num_patches) attention weights
            vmin: Minimum value for normalization (global mapping). If None, use image min.
            vmax: Maximum value for normalization (global mapping). If None, use image max.

        Returns:
            heatmap: (img_size, img_size, 3) RGB heatmap
            attention_resized: Resized attention map
            patch_attention: Raw patch attention values (for global min/max computation)
        """
        # Average attention across all patches (mean attention each patch receives)
        patch_attention = attention_map.mean(axis=0)  # (num_patches,)

        # Reshape to 2D grid
        attention_grid = patch_attention.reshape(
            self.num_patches_per_side,
            self.num_patches_per_side
        )

        # Normalize to [0, 1] using provided or computed min/max
        if vmin is None:
            vmin = attention_grid.min()
        if vmax is None:
            vmax = attention_grid.max()

        attention_grid = (attention_grid - vmin) / (vmax - vmin + 1e-8)
        attention_grid = np.clip(attention_grid, 0, 1)  # Ensure [0, 1] range

        # Resize to image size using bilinear interpolation
        attention_resized = cv2.resize(
            attention_grid,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply colormap (use 'jet' for better visualization)
        heatmap = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return heatmap, attention_resized, patch_attention

    def overlay_attention_on_image(self, image, attention_resized, alpha=0.5):
        """
        Overlay attention heatmap on original image.

        Args:
            image: Original image (H, W, 3)
            attention_resized: Attention map resized to image size (H, W)
            alpha: Blending factor

        Returns:
            overlaid: Blended image
        """
        # Create heatmap
        heatmap = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        return overlaid


def parse_image_condition(filename):
    """
    Parse condition from filename.
    Handles filenames like: base.jpg, expression1.jpg, eyeglasses1_facecover.jpg, etc.
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # Special case: base image
    if name_without_ext.lower() == 'base':
        return 'Base'

    # Condition label mapping
    condition_labels = {
        'base': 'Base',
        'expression': 'Expression',
        'eyeglasses': 'Eyeglasses',
        'facecover': 'Face Cover',
        'headcover': 'Head Cover',
        'scarf': 'Scarf',
        'illumination': 'Illumination',
        'occlusion': 'Occlusion',
        'pose': 'Pose',
    }

    # Split by underscore to handle combined conditions
    parts = name_without_ext.lower().split('_')

    # Build label from parts
    label_parts = []
    for part in parts:
        # Remove numbers for matching (e.g., "expression1" -> "expression")
        part_clean = ''.join([c for c in part if not c.isdigit()])

        # Check if it matches a known condition
        matched = False
        for key, label in condition_labels.items():
            if part_clean == key:
                # Keep the number if present
                number = ''.join([c for c in part if c.isdigit()])
                if number:
                    label_parts.append(f"{label} {number}")
                else:
                    label_parts.append(label)
                matched = True
                break

        if not matched and part:
            # Unknown part, just capitalize it
            label_parts.append(part.title())

    # Join with '+' for combined conditions
    if len(label_parts) > 1:
        return ' + '.join(label_parts)
    elif len(label_parts) == 1:
        return label_parts[0]
    else:
        # Fallback: use original filename
        return name_without_ext.replace('_', ' ').title()


def plot_attention_grid(visualizer, image_paths, output_file, title="ATTN-FIQA Attention Visualization", normalize_global=True):
    """
    Create grid plot with images, attention maps, and scores.

    Args:
        visualizer: AttentionVisualizer instance
        image_paths: List of image paths (up to 25)
        output_file: Output file path
        title: Plot title
        normalize_global: If True, use global min/max for colormap normalization across all images.
                         If False, normalize each image independently.
    """
    num_images = len(image_paths)

    # Determine grid size (5 columns)
    n_cols = 5
    n_rows = (num_images + n_cols - 1) // n_cols

    print(f"\nProcessing {num_images} images...")

    # First pass: extract all scores and attention values
    if normalize_global:
        print("  Pass 1: Extracting scores and attention for global normalization...")
    else:
        print("  Pass 1: Extracting scores and attention for per-image normalization...")

    all_scores = []
    all_attention_values = []
    all_intermediate_results = []

    for idx, image_path in enumerate(image_paths):
        print(f"    Processing image {idx+1}/{num_images}: {os.path.basename(image_path)}")

        try:
            image_tensor, original_image = visualizer.load_and_preprocess_image(image_path)
            score, attention_map = visualizer.extract_attention_and_score(image_tensor)

            # Store intermediate results
            all_scores.append(score)
            all_intermediate_results.append({
                'image_path': image_path,
                'original_image': original_image,
                'attention_map': attention_map,
                'score': score
            })

            # If global normalization, collect attention values
            if normalize_global:
                # Get raw patch attention for min/max computation
                patch_attention = attention_map.mean(axis=0)
                all_attention_values.extend(patch_attention.flatten())

        except Exception as e:
            print(f"      Error processing {image_path}: {e}")
            continue

    if len(all_scores) == 0:
        print("Error: No images were successfully processed!")
        return

    # Sort images by score (high to low)
    print(f"\n  Sorting images by quality score (high to low)...")
    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    all_intermediate_results = [all_intermediate_results[i] for i in sorted_indices]
    all_scores = [all_scores[i] for i in sorted_indices]

    print(f"    Highest score: {all_scores[0]:.6f} - {os.path.basename(all_intermediate_results[0]['image_path'])}")
    print(f"    Lowest score: {all_scores[-1]:.6f} - {os.path.basename(all_intermediate_results[-1]['image_path'])}")

    # Compute global attention range if needed
    attn_vmin = None
    attn_vmax = None
    if normalize_global and len(all_attention_values) > 0:
        attn_vmin = np.min(all_attention_values)
        attn_vmax = np.max(all_attention_values)
        print(f"\n  Global attention statistics:")
        print(f"    Min: {attn_vmin:.6f}")
        print(f"    Max: {attn_vmax:.6f}")

    # Normalize scores to [0, 1]
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score

    print(f"\n  Score statistics:")
    print(f"    Min: {min_score:.6f}")
    print(f"    Max: {max_score:.6f}")
    print(f"    Range: {score_range:.6f}")
    print(f"\n  Pass 2: Creating visualization (ordered by quality)...")

    # Create figure with 3 columns per image (original, attention, overlay)
    fig = plt.figure(figsize=(n_cols * 9, n_rows * 3.5))
    gs = gridspec.GridSpec(n_rows, n_cols * 3, figure=fig,
                          hspace=0.4, wspace=0.15,
                          left=0.05, right=0.95, top=0.93, bottom=0.05)

    # Second pass: create visualizations with proper normalization
    for idx, result in enumerate(all_intermediate_results):
        original_image = result['original_image']
        attention_map = result['attention_map']
        raw_score = result['score']
        image_path = result['image_path']

        # Create heatmap with global or per-image normalization
        heatmap, attention_resized, _ = visualizer.create_attention_heatmap(
            attention_map, vmin=attn_vmin, vmax=attn_vmax
        )
        overlaid = visualizer.overlay_attention_on_image(original_image, attention_resized, alpha=0.4)

        # Normalize score
        if score_range > 0:
            normalized_score = (raw_score - min_score) / score_range
        else:
            normalized_score = 0.5  # If all scores are the same

        # Determine position in grid
        row = idx // n_cols
        col_base = (idx % n_cols) * 3

        # Plot original image
        ax1 = fig.add_subplot(gs[row, col_base])
        ax1.imshow(original_image)
        ax1.axis('off')
        condition = parse_image_condition(image_path)
        ax1.set_title(f"{condition}", fontsize=11, fontweight='bold', pad=5)

        # Plot attention heatmap
        ax2 = fig.add_subplot(gs[row, col_base + 1])
        ax2.imshow(heatmap)
        ax2.axis('off')
        ax2.set_title(f"Attention Map", fontsize=11, fontweight='bold', pad=5)

        # Plot overlay with normalized score
        ax3 = fig.add_subplot(gs[row, col_base + 2])
        ax3.imshow(overlaid)
        ax3.axis('off')

        # Color code based on normalized score (green for high, red for low)
        if normalized_score >= 0.5:
            score_color = 'darkgreen'
        else:
            score_color = 'darkred'

        ax3.set_title(f"Score: {normalized_score:.3f}", fontsize=11, fontweight='bold',
                     pad=5, color=score_color)

    # Add main title
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"\nSaved visualization to:")
    print(f"  {output_path.with_suffix('.pdf')}")
    print(f"  {output_path.with_suffix('.png')}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize attention maps and quality scores for controlled face images'
    )

    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Directory containing images to visualize'
    )
    parser.add_argument(
        '--image-pattern',
        type=str,
        default='*.jpg',
        help='Pattern to match image files (e.g., "*.jpg", "*.png")'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='../results/attention_visualization/results.png',
        help='Output file path'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='../pretrained/',
        help='Path to directory containing pretrained model'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='vits_wf4m_adaface.pt',
        help='Name of the pretrained model file'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='vits',
        choices=['vits', 'vitb'],
        help='Backbone architecture (vits, vitb)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Attention Visualization',
        help='Plot title'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=25,
        help='Maximum number of images to visualize'
    )
    parser.add_argument(
        '--normalize-global',
        action='store_true',
        default=True,
        help='Use global min/max for colormap normalization (default: True)'
    )
    parser.add_argument(
        '--normalize-per-image',
        action='store_true',
        help='Use per-image min/max for colormap normalization (overrides --normalize-global)'
    )

    args = parser.parse_args()

    # Handle normalization flags
    if args.normalize_per_image:
        normalize_global = False
    else:
        normalize_global = args.normalize_global

    # Find images
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return

    import glob
    image_paths = sorted(glob.glob(str(image_dir / args.image_pattern)))

    if len(image_paths) == 0:
        print(f"Error: No images found matching pattern '{args.image_pattern}' in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Limit to max_images
    if len(image_paths) > args.max_images:
        print(f"Limiting to first {args.max_images} images")
        image_paths = image_paths[:args.max_images]

    # Initialize visualizer
    print(f"\nInitializing attention visualizer...")
    print(f"  Model: {args.model_name}")
    print(f"  Backbone: {args.backbone}")
    print(f"  GPU: {args.gpu_id}")

    visualizer = AttentionVisualizer(
        model_path=args.model_path,
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        backbone=args.backbone
    )

    # Create visualization
    print(f"  Colormap normalization: {'Global' if normalize_global else 'Per-image'}")

    plot_attention_grid(
        visualizer=visualizer,
        image_paths=image_paths,
        output_file=args.output_file,
        title=args.title,
        normalize_global=normalize_global
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
