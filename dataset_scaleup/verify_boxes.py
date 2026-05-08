import argparse
import json
import os
import random
from typing import Dict, Optional
from PIL import Image, ImageDraw

# Default output directory
DEFAULT_OUTPUT_DIR = "/project/llmsvgen/jinmin/SynLayers/visualization_output"


def load_sample_metadata(sample_dir: str) -> Optional[Dict]:
    """Load sample metadata."""
    metadata_path = os.path.join(sample_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_boxes(sample_dir: str, output_dir: str) -> Optional[str]:
    """
    Visualize bounding boxes on the whole image.
    
    Returns the output path if successful, None otherwise.
    """
    metadata = load_sample_metadata(sample_dir)
    if metadata is None:
        print(f"Failed to load metadata from {sample_dir}")
        return None
    
    whole_image_path = os.path.join(sample_dir, 'whole_image.png')
    if not os.path.exists(whole_image_path):
        print(f"whole_image.png not found in {sample_dir}")
        return None
    
    img = Image.open(whole_image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Colors for different sources
    colors = {
        'base': (0, 255, 0),           # Green
        'donor': (255, 0, 0),          # Red
        'laion_foreground': (0, 0, 255),  # Blue
        'caption': (255, 255, 0),      # Yellow
    }
    
    canvas_size = metadata.get('width', 512)
    
    for layer in metadata.get('layers', []):
        box = layer.get('box', [0, 0, 256, 256])
        source = layer.get('source', layer.get('type', 'unknown'))
        layer_idx = layer.get('layer_idx', -1)
        
        color = colors.get(source, (255, 255, 255))
        
        # Draw box
        draw.rectangle(box, outline=color, width=2)
        
        # Draw center point
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        r = 4
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=color)
        
        # Draw label
        label = f"L{layer_idx}"
        draw.text((box[0] + 4, box[1] + 4), label, fill=color)
    
    # Add legend at bottom
    legend_x = 10
    legend_y = canvas_size - 25
    for source, color in colors.items():
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color)
        draw.text((legend_x + 16, legend_y), source, fill=(255, 255, 255))
        legend_x += 100
    
    # Save to output directory
    sample_name = metadata.get('sample_dir', os.path.basename(sample_dir))
    output_path = os.path.join(output_dir, f"{sample_name}_boxes.png")
    img.save(output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes in scaled-up samples')
    parser.add_argument('--sample_dir', type=str, help='Path to a single sample directory')
    parser.add_argument('--dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    if args.sample_dir:
        # Visualize single sample
        output_path = visualize_boxes(args.sample_dir, args.output_dir)
        if output_path:
            print(f"✓ Saved: {output_path}")
    
    elif args.dataset_dir:
        # Visualize multiple samples
        sample_dirs = []
        for name in sorted(os.listdir(args.dataset_dir)):
            if name.startswith('sample_'):
                sample_dirs.append(os.path.join(args.dataset_dir, name))
        
        if len(sample_dirs) > args.num_samples:
            sample_dirs = random.sample(sample_dirs, args.num_samples)
        
        success_count = 0
        for sample_dir in sample_dirs:
            output_path = visualize_boxes(sample_dir, args.output_dir)
            if output_path:
                print(f"✓ Saved: {os.path.basename(output_path)}")
                success_count += 1
        
        print(f"\n=== Summary ===")
        print(f"Visualized: {success_count}/{len(sample_dirs)} samples")
        print(f"Output: {args.output_dir}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

