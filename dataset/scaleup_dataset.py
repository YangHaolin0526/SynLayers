import argparse
import json
import os
import random
import logging
import shutil
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from scaleup_utils import (
    load_jsonl,
    save_jsonl,
    load_blended_sample,
    get_blended_sample_dirs,
    compute_non_overlapping_box_xyxy,
    compute_total_overlap,
    create_layer_on_canvas,
    build_spatial_aware_caption,
    get_position_description,
    get_box_size,
    get_content_bbox,
    load_caption_list,
    get_laion_images_with_captions,
    get_caption_images_with_text,
    select_random_layers_from_samples,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default canvas size
CANVAS_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description='Scale up PrismLayersPro-blended dataset')
    parser.add_argument('--blended_dir', type=str, required=True,
                        help='Path to PrismLayersPro-blended directory')
    parser.add_argument('--laion_dir', type=str, required=True,
                        help='Path to LAION aesthetic images directory')
    parser.add_argument('--caption_dir', type=str, required=True,
                        help='Path to caption images directory')
    parser.add_argument('--caption_meta', type=str, required=True,
                        help='Path to captions.jsonl with caption text')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for scaled-up dataset')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of new samples to generate')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting sample index for output naming')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    # Layer selection parameters
    parser.add_argument('--min_donor_samples', type=int, default=2,
                        help='Minimum number of samples to pick layers from')
    parser.add_argument('--max_donor_samples', type=int, default=3,
                        help='Maximum number of samples to pick layers from')
    parser.add_argument('--min_layers_per_donor', type=int, default=1,
                        help='Minimum layers to pick from each donor sample')
    parser.add_argument('--max_layers_per_donor', type=int, default=2,
                        help='Maximum layers to pick from each donor sample')
    parser.add_argument('--added_layer_min_size', type=float, default=0.8,
                        help='Minimum size ratio for added layers')
    parser.add_argument('--added_layer_max_size', type=float, default=1.2,
                        help='Maximum size ratio for added layers')
    parser.add_argument('--laion_prob', type=float, default=0.1,
                        help='Probability of including LAION image layer')
    parser.add_argument('--caption_prob', type=float, default=0.2,
                        help='Probability of including caption text image layer')
    parser.add_argument('--laion_min_size', type=float, default=0.2,
                        help='Minimum size ratio for LAION layer')
    parser.add_argument('--laion_max_size', type=float, default=0.4,
                        help='Maximum size ratio for LAION layer')
    parser.add_argument('--caption_min_size', type=float, default=1.0,
                        help='Minimum size ratio for caption layer')
    parser.add_argument('--caption_max_size', type=float, default=1.2,
                        help='Maximum size ratio for caption layer')
    # Base layer removal parameters
    parser.add_argument('--min_layers_to_remove', type=int, default=2,
                        help='Minimum number of layers to remove from base sample')
    parser.add_argument('--max_layers_to_remove', type=int, default=4,
                        help='Maximum number of layers to remove from base sample')
    # AlphaVAE layer parameters
    parser.add_argument('--alphavae_dir', type=str, default=None,
                        help='Path to AlphaVAE_frontview images directory')
    parser.add_argument('--alphavae_prompts', type=str, default=None,
                        help='Path to prompts.txt for AlphaVAE captions')
    parser.add_argument('--alphavae_min_layers', type=int, default=0,
                        help='Minimum number of AlphaVAE layers to add per sample')
    parser.add_argument('--alphavae_max_layers', type=int, default=0,
                        help='Maximum number of AlphaVAE layers to add per sample')
    parser.add_argument('--alphavae_min_size', type=float, default=0.15,
                        help='Minimum size ratio for AlphaVAE layers')
    parser.add_argument('--alphavae_max_size', type=float, default=0.35,
                        help='Maximum size ratio for AlphaVAE layers')
    # Multiprocessing
    parser.add_argument('--max_base_samples', type=int, default=None,
                        help='Max number of base samples to use (sorted by name). '
                             'E.g. 18000 to use sample_000000..sample_017999')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip samples whose output directory already exists (for resuming)')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='Number of parallel workers (0 = single-process)')
    return parser.parse_args()


def create_scaled_up_sample(
    base_sample_dir: str,
    all_sample_dirs: List[str],
    laion_images: List[Tuple[str, str]],
    caption_images: List[Tuple[str, str]],
    alphavae_images: List[Tuple[str, str]],
    output_dir: str,
    sample_idx: int,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """
    Create a new scaled-up sample by combining a base sample with layers from other samples.
    
    Returns metadata dict or None if failed.
    """
    # Load base sample
    base_meta = load_blended_sample(base_sample_dir)
    if base_meta is None:
        logger.warning(f"Failed to load base sample: {base_sample_dir}")
        return None
    
    canvas_size = base_meta.get('width', CANVAS_SIZE)
    
    # Create output sample directory
    sample_name = f"sample_{sample_idx:06d}"
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Copy base_image
    base_image = base_meta.get('base_image')
    if base_image:
        base_image.save(os.path.join(sample_output_dir, 'base_image.png'))
    else:
        base_image = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        base_image.save(os.path.join(sample_output_dir, 'base_image.png'))
    
    # Start with base composite
    composite = base_image.copy()
    
    # Collect occupied boxes
    occupied_boxes = []
    
    # New layers list
    new_layers = []
    current_layer_idx = 0
    
    # === Step 1: Copy layers from base sample (excluding laion_foreground and caption types) ===
    # Also randomly remove 2-3 layers to keep total layer count reasonable
    base_layers = base_meta.get('layers', [])
    base_prism_layers = [l for l in base_layers if l.get('type') is None]
    
    # Randomly remove some layers from base
    num_to_remove = random.randint(args.min_layers_to_remove, args.max_layers_to_remove)
    num_to_remove = min(num_to_remove, max(0, len(base_prism_layers) - 1))  # Keep at least 1 layer
    
    removed_layer_indices = set()
    if num_to_remove > 0 and len(base_prism_layers) > 1:
        layers_to_remove = random.sample(base_prism_layers, num_to_remove)
        removed_layer_indices = {l['layer_idx'] for l in layers_to_remove}
    
    # Filter out removed layers
    base_prism_layers_filtered = [l for l in base_prism_layers if l['layer_idx'] not in removed_layer_indices]
    
    for layer in base_prism_layers_filtered:
        orig_layer_idx = layer['layer_idx']
        layer_img = base_meta.get('layer_images', {}).get(orig_layer_idx)
        
        if layer_img is None:
            continue
        
        orig_box = layer.get('box', [0, 0, canvas_size, canvas_size])
        caption = layer.get('caption', '')
        orig_w = orig_box[2] - orig_box[0]
        orig_h = orig_box[3] - orig_box[1]
        
        # Randomly reposition (layout-agnostic): find a non-overlapping spot
        best_box = None
        best_overlap_ratio = float('inf')
        new_box = None
        for _ in range(300):
            x0 = random.randint(0, max(0, canvas_size - orig_w))
            y0 = random.randint(0, max(0, canvas_size - orig_h))
            candidate = [x0, y0, x0 + orig_w, y0 + orig_h]
            box_area = orig_w * orig_h
            if box_area <= 0:
                continue
            overlap = compute_total_overlap(candidate, occupied_boxes)
            overlap_ratio = overlap / box_area
            if overlap == 0:
                new_box = candidate
                break
            if overlap_ratio < best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_box = candidate
        if new_box is None:
            new_box = best_box if best_box else [0, 0, orig_w, orig_h]
        
        # Place cropped layer onto full canvas at the new random position
        layer_canvas = create_layer_on_canvas(layer_img, new_box, canvas_size)
        
        # Save layer with new index
        layer_filename = f'layer_{current_layer_idx:02d}.png'
        layer_canvas.save(os.path.join(sample_output_dir, layer_filename))
        
        # Composite
        composite = Image.alpha_composite(composite, layer_canvas)
        
        # Record
        w, h = get_box_size(new_box)
        new_layers.append({
            'layer_idx': current_layer_idx,
            'caption': caption,
            'box': new_box,
            'width_dst': w,
            'height_dst': h,
            'image_path': layer_filename,
            'source': 'base',
            'source_sample': base_sample_dir,
        })
        
        occupied_boxes.append(new_box)
        current_layer_idx += 1
    
    # === Step 2: Add layers from other samples ===
    num_donors = random.randint(args.min_donor_samples, args.max_donor_samples)
    donor_layers = select_random_layers_from_samples(
        all_sample_dirs,
        exclude_sample=base_sample_dir,
        num_samples_to_pick=num_donors,
        num_layers_per_sample=(args.min_layers_per_donor, args.max_layers_per_donor)
    )
    
    for layer_img, layer_info, source_sample in donor_layers:
        caption = layer_info.get('caption', '')
        
        # Use the donor layer's original bounding box dimensions
        orig_box = layer_info.get('box', [0, 0, canvas_size, canvas_size])
        orig_w = orig_box[2] - orig_box[0]
        orig_h = orig_box[3] - orig_box[1]
        
        # Find a non-overlapping position for a box of this exact size
        best_box = None
        best_overlap_ratio = float('inf')
        new_box = None
        for _ in range(300):
            x0 = random.randint(0, max(0, canvas_size - orig_w))
            y0 = random.randint(0, max(0, canvas_size - orig_h))
            candidate = [x0, y0, x0 + orig_w, y0 + orig_h]
            box_area = orig_w * orig_h
            if box_area <= 0:
                continue
            overlap = compute_total_overlap(candidate, occupied_boxes)
            overlap_ratio = overlap / box_area
            if overlap == 0:
                new_box = candidate
                break
            if overlap_ratio < best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_box = candidate
        if new_box is None:
            new_box = best_box if best_box else [0, 0, orig_w, orig_h]
        
        # Create layer on canvas at new position
        layer_canvas = create_layer_on_canvas(layer_img, new_box, canvas_size)
        
        # Save layer
        layer_filename = f'layer_{current_layer_idx:02d}.png'
        layer_canvas.save(os.path.join(sample_output_dir, layer_filename))
        
        # Composite
        composite = Image.alpha_composite(composite, layer_canvas)
        
        # Record
        w, h = get_box_size(new_box)
        new_layers.append({
            'layer_idx': current_layer_idx,
            'caption': caption,
            'box': new_box,
            'width_dst': w,
            'height_dst': h,
            'image_path': layer_filename,
            'source': 'donor',
            'source_sample': source_sample,
            'original_layer_idx': layer_info.get('layer_idx'),
        })
        
        occupied_boxes.append(new_box)
        current_layer_idx += 1
    
    # === Step 2.5: Optionally add AlphaVAE layers (0 to max) ===
    if alphavae_images and args.alphavae_max_layers > 0:
        num_alpha = random.randint(args.alphavae_min_layers, args.alphavae_max_layers)
        if num_alpha > 0:
            selected_alpha = random.sample(alphavae_images, min(num_alpha, len(alphavae_images)))
            for alpha_path, alpha_caption in selected_alpha:
                try:
                    alpha_img = Image.open(alpha_path).convert('RGBA')
                except Exception as e:
                    logger.warning(f"Failed to load AlphaVAE image: {alpha_path}, {e}")
                    continue

                alpha_box = compute_non_overlapping_box_xyxy(
                    canvas_size, occupied_boxes,
                    min_size_ratio=args.alphavae_min_size,
                    max_size_ratio=args.alphavae_max_size,
                    max_attempts=300,
                    max_overlap_ratio=0.10,
                    center_margin=32
                )

                alpha_layer = create_layer_on_canvas(alpha_img, alpha_box, canvas_size)

                layer_filename = f'layer_{current_layer_idx:02d}.png'
                alpha_layer.save(os.path.join(sample_output_dir, layer_filename))

                composite = Image.alpha_composite(composite, alpha_layer)

                w, h = get_box_size(alpha_box)
                new_layers.append({
                    'layer_idx': current_layer_idx,
                    'caption': alpha_caption,
                    'box': alpha_box,
                    'width_dst': w,
                    'height_dst': h,
                    'image_path': layer_filename,
                    'type': 'alphavae',
                    'source_path': alpha_path,
                })

                occupied_boxes.append(alpha_box)
                current_layer_idx += 1

    # === Step 3: Optionally add LAION image ===
    laion_caption = None
    laion_path = None
    if random.random() < args.laion_prob and laion_images:
        laion_path, laion_caption = random.choice(laion_images)
        try:
            laion_img = Image.open(laion_path).convert('RGBA')
            laion_orig_size = laion_img.size
        except Exception as e:
            logger.warning(f"Failed to load LAION image: {laion_path}, {e}")
            laion_img = None
        
        if laion_img is not None:
            laion_box = compute_non_overlapping_box_xyxy(
                canvas_size, occupied_boxes,
                min_size_ratio=args.laion_min_size,
                max_size_ratio=args.laion_max_size,
                max_attempts=300,
                max_overlap_ratio=0.10,
                center_margin=32
            )
            
            # Create layer
            laion_layer = create_layer_on_canvas(laion_img, laion_box, canvas_size)
            
            # Save
            layer_filename = f'layer_{current_layer_idx:02d}.png'
            laion_layer.save(os.path.join(sample_output_dir, layer_filename))
            
            # Composite
            composite = Image.alpha_composite(composite, laion_layer)
            
            # Record
            w, h = get_box_size(laion_box)
            new_layers.append({
                'layer_idx': current_layer_idx,
                'caption': laion_caption,
                'box': laion_box,
                'width_dst': w,
                'height_dst': h,
                'image_path': layer_filename,
                'type': 'laion_foreground',
                'original_size': list(laion_orig_size),
            })
            
            occupied_boxes.append(laion_box)
            current_layer_idx += 1
    
    # === Step 4: Optionally add caption image ===
    caption_text = None
    caption_path = None
    if random.random() < args.caption_prob and caption_images:
        caption_path, caption_text = random.choice(caption_images)
        try:
            caption_img = Image.open(caption_path).convert('RGBA')
            caption_orig_size = caption_img.size
        except Exception as e:
            logger.warning(f"Failed to load caption image: {caption_path}, {e}")
            caption_img = None
        
        if caption_img is not None:
            caption_box = compute_non_overlapping_box_xyxy(
                canvas_size, occupied_boxes,
                min_size_ratio=args.caption_min_size,
                max_size_ratio=args.caption_max_size,
                max_attempts=300,
                max_overlap_ratio=0.10,
                center_margin=32
            )
            
            # Create layer
            caption_layer = create_layer_on_canvas(caption_img, caption_box, canvas_size)
            
            # Compute tight bbox from actual non-transparent content
            # Caption images have rectangular colored areas surrounded by transparency,
            # so we use the actual content bounds instead of the placement box.
            tight_box = get_content_bbox(caption_layer)
            if tight_box is not None:
                caption_box = tight_box
            
            # Save
            layer_filename = f'layer_{current_layer_idx:02d}.png'
            caption_layer.save(os.path.join(sample_output_dir, layer_filename))
            
            # Composite
            composite = Image.alpha_composite(composite, caption_layer)
            
            # Record
            w, h = get_box_size(caption_box)
            new_layers.append({
                'layer_idx': current_layer_idx,
                'caption': f"Text: {caption_text}" if caption_text else "Text",
                'box': caption_box,
                'width_dst': w,
                'height_dst': h,
                'image_path': layer_filename,
                'type': 'caption',
                'original_size': list(caption_orig_size),
            })
            
            occupied_boxes.append(caption_box)
            current_layer_idx += 1
    
    # === Step 5: Save whole_image ===
    composite.save(os.path.join(sample_output_dir, 'whole_image.png'))
    
    # === Step 6: Build spatial-aware whole caption ===
    base_caption = base_meta.get('base_caption', '')
    whole_caption = build_spatial_aware_caption(new_layers, canvas_size, base_caption)
    
    # === Step 7: Create metadata ===
    metadata = {
        'id': f'{sample_idx:09d}',
        'style_category': base_meta.get('style_category', ''),
        'whole_caption': whole_caption,
        'base_caption': base_caption,
        'layer_count': len(new_layers),
        'layers': new_layers,
        # Extra fields
        'sample_dir': sample_name,
        'width': canvas_size,
        'height': canvas_size,
        'base_sample': base_sample_dir,
        'num_base_layers_removed': len(removed_layer_indices),
        'num_donor_samples': num_donors,
        'num_donated_layers': len(donor_layers),
    }
    
    if laion_path:
        metadata['laion_path'] = laion_path
        metadata['laion_caption'] = laion_caption
    if caption_path:
        metadata['caption_path'] = caption_path
        metadata['caption_text'] = caption_text
    
    # Save metadata
    with open(os.path.join(sample_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata


def _worker_fn(task):
    """Worker function for multiprocessing. Each task is a dict with all needed info."""
    sample_idx = task['sample_idx']
    base_sample_dir = task['base_sample_dir']
    all_sample_dirs = task['all_sample_dirs']
    laion_images = task['laion_images']
    caption_images = task['caption_images']
    alphavae_images = task['alphavae_images']
    output_dir = task['output_dir']
    args = task['args']
    seed = task['seed']

    if args.skip_existing:
        meta_path = os.path.join(output_dir, f"sample_{sample_idx:06d}", 'metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass

    random.seed(seed)

    try:
        metadata = create_scaled_up_sample(
            base_sample_dir=base_sample_dir,
            all_sample_dirs=all_sample_dirs,
            laion_images=laion_images,
            caption_images=caption_images,
            alphavae_images=alphavae_images,
            output_dir=output_dir,
            sample_idx=sample_idx,
            args=args,
        )
        return metadata
    except Exception as e:
        logger.error(f"Error processing sample {sample_idx}: {e}")
        return None


def main():
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load existing blended samples
    logger.info("Loading existing blended samples...")
    all_sample_dirs = get_blended_sample_dirs(args.blended_dir, max_samples=args.max_base_samples)
    logger.info(f"Found {len(all_sample_dirs)} existing samples"
                + (f" (limited to first {args.max_base_samples})" if args.max_base_samples else ""))

    if len(all_sample_dirs) < 10:
        logger.error("Not enough existing samples to create scaled-up dataset!")
        return

    # Load caption list
    logger.info("Loading caption list from captions.jsonl...")
    caption_list = load_caption_list(args.caption_meta)
    logger.info(f"Loaded {len(caption_list)} caption entries")

    # Load LAION images (cap at 20000 for balanced diversity)
    logger.info("Loading LAION images with captions...")
    laion_images = get_laion_images_with_captions(args.laion_dir)
    if len(laion_images) > 20000:
        random.shuffle(laion_images)
        laion_images = laion_images[:20000]
    logger.info(f"Using {len(laion_images)} LAION images")

    # Load caption images
    logger.info("Loading caption images...")
    caption_images = get_caption_images_with_text(args.caption_dir, caption_list)
    logger.info(f"Found {len(caption_images)} caption images")

    # Load AlphaVAE images (optional)
    alphavae_images = []  # list of (image_path, caption)
    if args.alphavae_dir and args.alphavae_prompts:
        logger.info("Loading AlphaVAE images with prompts...")
        with open(args.alphavae_prompts, 'r') as f:
            alphavae_prompts = [l.strip() for l in f.readlines() if l.strip()]
        alpha_files = sorted([
            f for f in os.listdir(args.alphavae_dir)
            if f.endswith('.png')
        ])
        for fname in alpha_files:
            idx = int(fname.replace('.png', ''))
            prompt_idx = idx // 5
            if prompt_idx < len(alphavae_prompts):
                caption = alphavae_prompts[prompt_idx]
            else:
                caption = ""
            alphavae_images.append((os.path.join(args.alphavae_dir, fname), caption))
        logger.info(f"Found {len(alphavae_images)} AlphaVAE images")

    # Pre-generate tasks with deterministic per-sample seeds
    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.num_samples):
        sample_idx = args.start_index + i
        tasks.append({
            'sample_idx': sample_idx,
            'base_sample_dir': rng.choice(all_sample_dirs),
            'all_sample_dirs': all_sample_dirs,
            'laion_images': laion_images,
            'caption_images': caption_images,
            'alphavae_images': alphavae_images,
            'output_dir': args.output_dir,
            'args': args,
            'seed': rng.randint(0, 2**31),
        })

    # Generate samples
    all_metadata = []
    failed_count = 0
    num_workers = args.num_workers

    if num_workers > 0:
        logger.info(f"Using multiprocessing with {num_workers} workers")
        with Pool(processes=num_workers) as pool:
            for metadata in tqdm(
                pool.imap_unordered(_worker_fn, tasks),
                total=len(tasks),
                desc="Generating samples"
            ):
                if metadata:
                    all_metadata.append(metadata)
                else:
                    failed_count += 1
    else:
        logger.info("Using single-process mode")
        for task in tqdm(tasks, desc="Generating samples"):
            metadata = _worker_fn(task)
            if metadata:
                all_metadata.append(metadata)
            else:
                failed_count += 1

    # Sort by sample index for deterministic output order
    all_metadata.sort(key=lambda m: int(m['id']))

    # Save index (scaleup_meta.jsonl)
    index_path = os.path.join(args.output_dir, 'scaleup_meta.jsonl')
    with open(index_path, 'w', encoding='utf-8') as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    logger.info(f"Generated {len(all_metadata)} samples ({failed_count} failed)")
    logger.info(f"Output saved to {args.output_dir}")
    logger.info(f"Index saved to {index_path}")


if __name__ == '__main__':
    main()

