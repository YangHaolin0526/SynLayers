"""
Utility functions for scaling up the PrismLayersPro-blended dataset.

This module provides utilities for:
- Loading existing blended samples
- Computing non-overlapping bounding boxes
- Generating spatial-aware captions with position words
- Layer combination and compositing
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], path: str):
    """Save list of dictionaries to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_blended_sample(sample_dir: str) -> Optional[Dict]:
    """
    Load a blended sample from its directory.
    Returns metadata dict with loaded layer images.
    """
    metadata_path = os.path.join(sample_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load base_image (background)
    base_path = os.path.join(sample_dir, 'base_image.png')
    if os.path.exists(base_path):
        metadata['base_image'] = Image.open(base_path).convert('RGBA')
    else:
        metadata['base_image'] = None
    
    # Load layer images
    metadata['layer_images'] = {}
    for layer in metadata.get('layers', []):
        img_path = os.path.join(sample_dir, layer['image_path'])
        if os.path.exists(img_path):
            metadata['layer_images'][layer['layer_idx']] = Image.open(img_path).convert('RGBA')
    
    # Store sample directory path
    metadata['sample_path'] = sample_dir
    
    return metadata


def get_blended_sample_dirs(blended_dir: str, max_samples: Optional[int] = None) -> List[str]:
    """
    Get list of sample directories in the blended directory.
    """
    sample_dirs = []
    for name in sorted(os.listdir(blended_dir)):
        if name.startswith('sample_') and os.path.isdir(os.path.join(blended_dir, name)):
            sample_dirs.append(os.path.join(blended_dir, name))
            if max_samples and len(sample_dirs) >= max_samples:
                break
    return sample_dirs


def compute_overlap_area(box1: List[int], box2: List[int]) -> int:
    """
    Calculate the overlap area between two boxes (xyxy format).
    Returns 0 if no overlap.
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    # Check if there's an intersection
    if x0_i >= x1_i or y0_i >= y1_i:
        return 0
    
    return (x1_i - x0_i) * (y1_i - y0_i)


def compute_total_overlap(box: List[int], existing_boxes: List[List[int]]) -> int:
    """
    Calculate total overlap area between a box and all existing boxes.
    """
    total = 0
    for eb in existing_boxes:
        total += compute_overlap_area(box, eb)
    return total


def get_position_description(box: List[int], canvas_size: int) -> str:
    """
    Get position description for a bounding box.
    
    Based on the box center point position, returns one of:
    - "On the top-left"
    - "On the top-right"
    - "On the bottom-left"
    - "On the bottom-right"
    - "In the center"
    - "At the top"
    - "At the bottom"
    - "On the left"
    - "On the right"
    """
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    
    # Normalize to 0-1 range
    norm_x = center_x / canvas_size
    norm_y = center_y / canvas_size
    
    # Define regions (3x3 grid)
    # Left: 0-0.33, Center: 0.33-0.67, Right: 0.67-1.0
    # Top: 0-0.33, Middle: 0.33-0.67, Bottom: 0.67-1.0
    
    if norm_y < 0.33:
        if norm_x < 0.33:
            return "On the top-left"
        elif norm_x > 0.67:
            return "On the top-right"
        else:
            return "At the top"
    elif norm_y > 0.67:
        if norm_x < 0.33:
            return "On the bottom-left"
        elif norm_x > 0.67:
            return "On the bottom-right"
        else:
            return "At the bottom"
    else:
        if norm_x < 0.33:
            return "On the left"
        elif norm_x > 0.67:
            return "On the right"
        else:
            return "In the center"


def build_spatial_aware_caption(layers: List[Dict], canvas_size: int, base_caption: str = "") -> str:
    """
    Build a spatial-aware whole caption by adding position descriptions to each layer.
    
    Example output:
    "On the top-left, a red balloon. In the center, a clown character. At the bottom, Text: hello world."
    
    This structured format with spatial information helps diffusion models (especially Flux with T5)
    better understand the position-layer correspondence.
    """
    parts = []
    
    # Add base caption if provided (shortened version)
    if base_caption:
        # Take only the first sentence of base caption to keep it concise
        first_sentence = base_caption.split('.')[0].strip()
        if first_sentence:
            parts.append(first_sentence + ".")
    
    # Add layer descriptions with position
    for layer in layers:
        caption = layer.get('caption', '').strip()
        if not caption:
            continue
        
        box = layer.get('box', [0, 0, canvas_size, canvas_size])
        position = get_position_description(box, canvas_size)
        
        # Clean up caption - remove leading "The picture/image features" etc.
        caption_clean = caption
        prefixes_to_remove = [
            "The picture features ",
            "The image features ",
            "Text ",
        ]
        for prefix in prefixes_to_remove:
            if caption_clean.startswith(prefix):
                caption_clean = caption_clean[len(prefix):]
                break
        
        # Capitalize first letter
        if caption_clean:
            caption_clean = caption_clean[0].upper() + caption_clean[1:] if len(caption_clean) > 1 else caption_clean.upper()
        
        # Remove trailing period if present
        caption_clean = caption_clean.rstrip('.')
        
        parts.append(f"{position}, {caption_clean}.")
    
    return " ".join(parts)


def compute_random_box_xyxy(
    canvas_size: int,
    min_size_ratio: float = 0.10,
    max_size_ratio: float = 0.25,
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
    center_margin: int = 16
) -> List[int]:
    """
    Compute a random bounding box in xyxy format [x0, y0, x1, y1].
    
    Args:
        canvas_size: Size of the canvas (e.g., 512)
        min_size_ratio: Minimum size as ratio of canvas
        max_size_ratio: Maximum size as ratio of canvas
        aspect_ratio_range: Range of aspect ratios (width/height)
        center_margin: Margin from edges for box center (e.g., 16 means center 
                       must be within [16, canvas_size-16] range, i.e., 480x480 area for 512 canvas)
    """
    min_size = int(canvas_size * min_size_ratio)
    max_size = int(canvas_size * max_size_ratio)
    
    # Random aspect ratio
    aspect_ratio = random.uniform(*aspect_ratio_range)
    
    if aspect_ratio >= 1.0:
        w = random.randint(min_size, max_size)
        h = int(w / aspect_ratio)
    else:
        h = random.randint(min_size, max_size)
        w = int(h * aspect_ratio)
    
    # Clamp to valid range
    w = max(min_size, min(w, max_size))
    h = max(min_size, min(h, max_size))
    
    # Random center position within the allowed region (canvas_size - 2*margin)
    # For 512 canvas with margin=16, center can be in [16, 496]
    min_center = center_margin
    max_center = canvas_size - center_margin
    
    # Ensure we have valid range
    if max_center <= min_center:
        max_center = canvas_size - 1
        min_center = 0
    
    center_x = random.randint(min_center, max_center)
    center_y = random.randint(min_center, max_center)
    
    # Convert center to box coordinates
    x0 = center_x - w // 2
    y0 = center_y - h // 2
    x1 = x0 + w
    y1 = y0 + h
    
    # Clamp to canvas bounds (box can extend to edges, just center is constrained)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(canvas_size, x1)
    y1 = min(canvas_size, y1)
    
    return [x0, y0, x1, y1]


def compute_non_overlapping_box_xyxy(
    canvas_size: int,
    existing_boxes: List[List[int]],
    min_size_ratio: float = 0.10,
    max_size_ratio: float = 0.25,
    max_attempts: int = 300,
    max_overlap_ratio: float = 0.20,
    center_margin: int = 16
) -> List[int]:
    """
    Compute a box (xyxy) that minimizes overlap with existing boxes.
    
    Args:
        canvas_size: Size of the canvas (e.g., 512)
        existing_boxes: List of existing boxes to avoid overlapping with
        min_size_ratio: Minimum size as ratio of canvas
        max_size_ratio: Maximum size as ratio of canvas
        max_attempts: Maximum attempts to find a good position
        max_overlap_ratio: Maximum acceptable overlap ratio (default 20%)
        center_margin: Margin from edges for box center (default 16px, so center
                       is within 480x480 area for 512 canvas)
    
    Strategy:
    1. Try to find a position with no overlap
    2. If not possible, accept positions with < max_overlap_ratio overlap
    3. Return the position with minimum overlap
    """
    best_box = None
    best_overlap_ratio = float('inf')
    
    for _ in range(max_attempts):
        box = compute_random_box_xyxy(
            canvas_size, min_size_ratio, max_size_ratio,
            center_margin=center_margin
        )
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        if box_area <= 0:
            continue
        
        overlap = compute_total_overlap(box, existing_boxes)
        overlap_ratio = overlap / box_area
        
        # If no overlap, return immediately
        if overlap == 0:
            return box
        
        # Track best box
        if overlap_ratio < best_overlap_ratio:
            best_overlap_ratio = overlap_ratio
            best_box = box
        
        # Accept if overlap is small enough
        if overlap_ratio < max_overlap_ratio:
            return box
    
    # Return the best box found
    if best_box is not None:
        return best_box
    
    # Fallback
    return compute_random_box_xyxy(
        canvas_size, min_size_ratio, max_size_ratio,
        center_margin=center_margin
    )


def create_layer_on_canvas(
    layer_img: Image.Image,
    box: List[int],
    canvas_size: int
) -> Image.Image:
    """
    Create a full-canvas RGBA image with the layer placed at box position.
    Box is in xyxy format: [x0, y0, x1, y1].
    Layer will have transparent background.
    """
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    
    # Create transparent canvas
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    
    # Ensure positive dimensions
    if w <= 0 or h <= 0:
        return canvas
    
    # Resize layer to fit box
    layer_resized = layer_img.resize((w, h), Image.LANCZOS)
    
    # Paste with alpha (preserving transparency)
    if layer_resized.mode == 'RGBA':
        canvas.paste(layer_resized, (x0, y0), layer_resized)
    else:
        layer_resized = layer_resized.convert('RGBA')
        canvas.paste(layer_resized, (x0, y0), layer_resized)
    
    return canvas


def get_content_bbox(img: Image.Image) -> Optional[List[int]]:
    """
    Get the tight bounding box of non-transparent content in an RGBA image.
    Returns [x0, y0, x1, y1] or None if the image is fully transparent.
    """
    arr = np.array(img.convert('RGBA'))
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax + 1), int(rmax + 1)]


def get_box_size(box: List[int]) -> Tuple[int, int]:
    """Get width and height from xyxy box."""
    x0, y0, x1, y1 = box
    return (x1 - x0, y1 - y0)


def load_caption_list(caption_jsonl: str) -> List[Dict]:
    """
    Load captions.jsonl as a list (ordered by line number).
    """
    return load_jsonl(caption_jsonl)


def get_laion_caption_from_json(image_path: str) -> str:
    """
    Get LAION image caption from its corresponding .json file.
    """
    json_path = image_path.rsplit('.', 1)[0] + '.json'
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('caption', '')
        except Exception:
            pass
    
    return os.path.basename(image_path).rsplit('.', 1)[0]


def get_laion_images_with_captions(laion_dir: str, laion_jsonl: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get all LAION images with their captions.
    """
    images = []
    
    for subdir in sorted(os.listdir(laion_dir)):
        subdir_path = os.path.join(laion_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in sorted(os.listdir(subdir_path)):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(subdir_path, fname)
                    caption = get_laion_caption_from_json(img_path)
                    images.append((img_path, caption))
    
    return images


def get_caption_images_with_text(caption_dir: str, caption_list: List[Dict]) -> List[Tuple[str, str]]:
    """
    Get caption images with their text content.
    """
    images = []
    
    for fname in sorted(os.listdir(caption_dir)):
        if fname.endswith('.png'):
            img_path = os.path.join(caption_dir, fname)
            
            idx_str = fname.split('.')[0]
            try:
                idx = int(idx_str)
            except ValueError:
                idx = -1
            
            caption_text = ""
            if 0 <= idx < len(caption_list):
                caption_text = caption_list[idx].get('caption', '')
            
            images.append((img_path, caption_text))
    
    return images


def extract_layer_from_sample(
    sample_metadata: Dict,
    layer_idx: int
) -> Optional[Tuple[Image.Image, Dict]]:
    """
    Extract a specific layer from a sample.
    Returns (layer_image, layer_info) or None if not found.
    """
    layer_images = sample_metadata.get('layer_images', {})
    
    if layer_idx not in layer_images:
        return None
    
    # Find layer info
    for layer in sample_metadata.get('layers', []):
        if layer['layer_idx'] == layer_idx:
            return (layer_images[layer_idx], layer.copy())
    
    return None


def select_random_layers_from_samples(
    sample_dirs: List[str],
    exclude_sample: str,
    num_samples_to_pick: int = 2,
    num_layers_per_sample: Tuple[int, int] = (1, 2)
) -> List[Tuple[Image.Image, Dict, str]]:
    """
    Select random layers from random samples.
    
    Args:
        sample_dirs: List of all sample directories
        exclude_sample: Sample directory to exclude (the base sample)
        num_samples_to_pick: Number of different samples to pick from (2-3)
        num_layers_per_sample: Range of layers to pick from each sample (min, max)
    
    Returns:
        List of (layer_image, layer_info, source_sample) tuples
    """
    # Filter out the base sample
    available_samples = [s for s in sample_dirs if s != exclude_sample]
    
    if len(available_samples) < num_samples_to_pick:
        num_samples_to_pick = len(available_samples)
    
    # Randomly select samples
    selected_samples = random.sample(available_samples, num_samples_to_pick)
    
    collected_layers = []
    
    for sample_dir in selected_samples:
        # Load sample
        sample_meta = load_blended_sample(sample_dir)
        if sample_meta is None:
            continue
        
        # Get available layers (excluding laion_foreground and caption types to avoid duplicates)
        layers = sample_meta.get('layers', [])
        prism_layers = [l for l in layers if l.get('type') is None]  # Original prism layers only
        
        if not prism_layers:
            continue
        
        # Randomly select how many layers to pick
        min_layers, max_layers = num_layers_per_sample
        num_to_pick = random.randint(min_layers, min(max_layers, len(prism_layers)))
        
        # Select random layers
        selected_layers = random.sample(prism_layers, num_to_pick)
        
        for layer_info in selected_layers:
            layer_idx = layer_info['layer_idx']
            layer_img = sample_meta.get('layer_images', {}).get(layer_idx)
            
            if layer_img is not None:
                collected_layers.append((layer_img, layer_info.copy(), sample_dir))
    
    return collected_layers

