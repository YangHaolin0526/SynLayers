import argparse
import json
import os
import random
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

from blend_utils import (
    load_jsonl,
    load_prism_sample,
    get_prism_sample_dirs,
    scale_box_xyxy,
    compute_non_overlapping_box_xyxy,
    create_layer_on_canvas,
    build_whole_caption,
    get_box_size,
    load_caption_list,
    get_laion_images_with_captions,
    get_caption_images_with_text,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PrismLayersPro original size
PRISM_ORIGINAL_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description='Blend PrismLayersPro with LAION and caption images')
    parser.add_argument('--prism_dir', type=str, required=True,
                        help='Path to PrismLayersPro-image directory')
    parser.add_argument('--laion_dir', type=str, required=True,
                        help='Path to LAION aesthetic images directory')
    parser.add_argument('--laion_jsonl', type=str, default=None,
                        help='Optional JSONL with LAION image captions')
    parser.add_argument('--caption_dir', type=str, required=True,
                        help='Path to caption images directory')
    parser.add_argument('--caption_meta', type=str, required=True,
                        help='Path to captions.jsonl with caption text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for blended dataset')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Output image size (default: 512)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting sample index (for creating test set)')
    parser.add_argument('--max_samples', type=int, default=19500,
                        help='Maximum number of samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--laion_min_size', type=float, default=0.2,
                        help='Minimum size ratio for LAION layer (35% of image)')
    parser.add_argument('--laion_max_size', type=float, default=0.4,
                        help='Maximum size ratio for LAION layer (60% of image)')
    parser.add_argument('--caption_min_size', type=float, default=0.5,
                        help='Minimum size ratio for caption layer (30% of image)')
    parser.add_argument('--caption_max_size', type=float, default=0.6,
                        help='Maximum size ratio for caption layer (55% of image)')
    return parser.parse_args()


def get_occupied_boxes_scaled(prism_metadata: Dict, scale: float) -> List[List[int]]:
    """
    Extract occupied boxes from prism layers and scale them.
    Boxes are in xyxy format: [x0, y0, x1, y1].
    """
    boxes = []
    for layer in prism_metadata.get('layers', []):
        box = layer.get('box')
        if box and len(box) == 4:
            # Scale box from 1024 to target size
            scaled = [int(b * scale) for b in box]
            boxes.append(scaled)
    return boxes


def process_sample(
    prism_sample_dir: str,
    laion_images: List[Tuple[str, str]],
    caption_images: List[Tuple[str, str]],
    output_dir: str,
    sample_idx: int,
    output_size: int,
    laion_min_size: float,
    laion_max_size: float,
    caption_min_size: float,
    caption_max_size: float,
) -> Optional[Dict]:
    """
    Process a single PrismLayersPro sample.
    Returns metadata dict or None if failed.
    """
    # Load prism sample
    prism_metadata = load_prism_sample(prism_sample_dir)
    if not prism_metadata:
        logger.warning(f"Failed to load prism sample: {prism_sample_dir}")
        return None
    
    # Scale factor from original (1024) to target size
    scale = output_size / PRISM_ORIGINAL_SIZE
    
    # Create output sample directory
    sample_name = f"sample_{sample_idx:06d}"
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Get occupied boxes from prism layers (scaled)
    occupied_boxes = get_occupied_boxes_scaled(prism_metadata, scale)
    
    # Start building layers list (following PrismLayersPro convention)
    new_layers = []
    prism_layer_count = prism_metadata.get('layer_count', 0)
    
    # Process and save base_image (background)
    base_image = prism_metadata.get('base_image')
    if base_image:
        base_image_scaled = base_image.resize((output_size, output_size), Image.LANCZOS)
        base_image_scaled.save(os.path.join(sample_output_dir, 'base_image.png'))
    else:
        # Create transparent base if not found
        base_image_scaled = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
        base_image_scaled.save(os.path.join(sample_output_dir, 'base_image.png'))
    
    # Process prism layers
    # Start with base (transparent canvas for compositing)
    composite = base_image_scaled.copy()
    
    for layer in prism_metadata.get('layers', []):
        layer_idx = layer['layer_idx']
        layer_caption = layer.get('caption', '')
        box = layer.get('box', [0, 0, PRISM_ORIGINAL_SIZE, PRISM_ORIGINAL_SIZE])
        
        # Scale box from 1024 to target size
        scaled_box = scale_box_xyxy(box, PRISM_ORIGINAL_SIZE, output_size)
        
        # Load layer image
        layer_img = prism_metadata['layer_images'].get(layer_idx)
        if layer_img is None:
            continue
        
        # Get target dimensions from scaled box
        w = scaled_box[2] - scaled_box[0]
        h = scaled_box[3] - scaled_box[1]
        
        # Create layer on full canvas (with transparency)
        layer_canvas = create_layer_on_canvas(layer_img, scaled_box, output_size)
        
        # Save layer image (following PrismLayersPro naming: layer_00.png, layer_01.png, ...)
        layer_filename = f'layer_{layer_idx:02d}.png'
        layer_canvas.save(os.path.join(sample_output_dir, layer_filename))
        
        # Composite onto the merged image
        composite = Image.alpha_composite(composite, layer_canvas)
        
        new_layers.append({
            'layer_idx': layer_idx,
            'caption': layer_caption,
            'box': scaled_box,  # xyxy format
            'width_dst': w,
            'height_dst': h,
            'image_path': layer_filename
        })
    
    # === Add LAION image layer ===
    laion_path, laion_caption = random.choice(laion_images)
    try:
        laion_img = Image.open(laion_path).convert('RGBA')
        laion_orig_size = laion_img.size
    except Exception as e:
        logger.warning(f"Failed to load LAION image: {laion_path}, {e}")
        laion_img = Image.new('RGBA', (256, 256), (128, 128, 128, 255))
        laion_orig_size = (256, 256)
        laion_caption = "image"
    
    # Find non-overlapping position for LAION layer
    laion_box = compute_non_overlapping_box_xyxy(
        output_size, occupied_boxes,
        min_size_ratio=laion_min_size,
        max_size_ratio=laion_max_size
    )
    occupied_boxes.append(laion_box)
    
    # Create LAION layer on canvas (with transparency)
    laion_layer = create_layer_on_canvas(laion_img, laion_box, output_size)
    
    # Next layer index
    laion_layer_idx = prism_layer_count
    laion_filename = f'layer_{laion_layer_idx:02d}.png'
    laion_layer.save(os.path.join(sample_output_dir, laion_filename))
    
    # Composite
    composite = Image.alpha_composite(composite, laion_layer)
    
    laion_w, laion_h = get_box_size(laion_box)
    new_layers.append({
        'layer_idx': laion_layer_idx,
        'caption': laion_caption,
        'box': laion_box,  # xyxy format
        'width_dst': laion_w,
        'height_dst': laion_h,
        'image_path': laion_filename,
        'type': 'laion_foreground',
        'original_size': list(laion_orig_size)
    })
    
    # === Add caption image layer ===
    caption_path, caption_text = random.choice(caption_images)
    try:
        caption_img = Image.open(caption_path).convert('RGBA')
        caption_orig_size = caption_img.size
    except Exception as e:
        logger.warning(f"Failed to load caption image: {caption_path}, {e}")
        caption_img = Image.new('RGBA', (256, 128), (255, 255, 255, 255))
        caption_orig_size = (256, 128)
        caption_text = "text"
    
    # Find non-overlapping position for caption layer
    caption_box = compute_non_overlapping_box_xyxy(
        output_size, occupied_boxes,
        min_size_ratio=caption_min_size,
        max_size_ratio=caption_max_size
    )
    
    # Create caption layer on canvas (with transparency)
    caption_layer = create_layer_on_canvas(caption_img, caption_box, output_size)
    
    # Next layer index
    caption_layer_idx = prism_layer_count + 1
    caption_filename = f'layer_{caption_layer_idx:02d}.png'
    caption_layer.save(os.path.join(sample_output_dir, caption_filename))
    
    # Composite
    composite = Image.alpha_composite(composite, caption_layer)
    
    caption_w, caption_h = get_box_size(caption_box)
    new_layers.append({
        'layer_idx': caption_layer_idx,
        'caption': f"Text: {caption_text}" if caption_text else "Text",
        'box': caption_box,  # xyxy format
        'width_dst': caption_w,
        'height_dst': caption_h,
        'image_path': caption_filename,
        'type': 'caption',
        'original_size': list(caption_orig_size)
    })
    
    # Save whole_image (final composite)
    composite.save(os.path.join(sample_output_dir, 'whole_image.png'))
    
    # Build whole caption
    prism_caption = prism_metadata.get('whole_caption', '')
    whole_caption = build_whole_caption(prism_caption, laion_caption, caption_text)
    
    # Create metadata (following PrismLayersPro format)
    metadata = {
        'id': f'{sample_idx:09d}',
        'style_category': prism_metadata.get('style_category', ''),
        'whole_caption': whole_caption,
        'base_caption': prism_metadata.get('base_caption', ''),
        'layer_count': len(new_layers),
        'layers': new_layers,
        # Extra fields for our blended dataset
        'sample_dir': sample_name,
        'width': output_size,
        'height': output_size,
        'source_prism_dir': prism_sample_dir,
        'laion_path': laion_path,
        'laion_caption': laion_caption,
        'caption_path': caption_path,
        'caption_text': caption_text,
    }
    
    # Save metadata
    with open(os.path.join(sample_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


def main():
    args = parse_args()
    random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get prism sample directories
    logger.info("Loading PrismLayersPro samples...")
    prism_samples = get_prism_sample_dirs(args.prism_dir)
    
    if args.start_index > 0:
        prism_samples = prism_samples[args.start_index:]
    
    if args.max_samples:
        prism_samples = prism_samples[:args.max_samples]
    
    logger.info(f"Found {len(prism_samples)} prism samples to process")
    
    # Load caption list (for caption images)
    logger.info("Loading caption list from captions.jsonl...")
    caption_list = load_caption_list(args.caption_meta)
    logger.info(f"Loaded {len(caption_list)} caption entries")
    
    # Load LAION images (caption from .json files next to each image)
    logger.info("Loading LAION images with captions from .json files...")
    laion_images = get_laion_images_with_captions(args.laion_dir, args.laion_jsonl)
    logger.info(f"Found {len(laion_images)} LAION images")
    
    # Load caption images (caption text from captions.jsonl by index)
    logger.info("Loading caption images...")
    caption_images = get_caption_images_with_text(args.caption_dir, caption_list)
    logger.info(f"Found {len(caption_images)} caption images")
    
    if not laion_images:
        logger.error("No LAION images found!")
        return
    if not caption_images:
        logger.error("No caption images found!")
        return
    
    # Process samples
    all_metadata = []
    for i, prism_sample_dir in enumerate(tqdm(prism_samples, desc="Processing samples")):
        sample_idx = args.start_index + i
        
        metadata = process_sample(
            prism_sample_dir=prism_sample_dir,
            laion_images=laion_images,
            caption_images=caption_images,
            output_dir=args.output_dir,
            sample_idx=sample_idx,
            output_size=args.output_size,
            laion_min_size=args.laion_min_size,
            laion_max_size=args.laion_max_size,
            caption_min_size=args.caption_min_size,
            caption_max_size=args.caption_max_size,
        )
        
        if metadata:
            all_metadata.append(metadata)
    
    # Save index (blend_meta.jsonl)
    index_path = os.path.join(args.output_dir, 'blend_meta.jsonl')
    with open(index_path, 'w', encoding='utf-8') as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(all_metadata)} samples to {args.output_dir}")
    logger.info(f"Index saved to {index_path}")


if __name__ == '__main__':
    main()
