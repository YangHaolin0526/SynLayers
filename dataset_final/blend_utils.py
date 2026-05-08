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


def load_prism_sample(sample_dir: str) -> Optional[Dict]:
    """
    Load a PrismLayersPro sample from its directory.
    Returns metadata dict with loaded images.
    
    Expected files:
    - metadata.json
    - whole_image.png
    - base_image.png
    - layer_00.png, layer_01.png, ...
    """
    metadata_path = os.path.join(sample_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load whole_image
    whole_path = os.path.join(sample_dir, 'whole_image.png')
    if os.path.exists(whole_path):
        metadata['whole_image'] = Image.open(whole_path).convert('RGBA')
    
    # Load base_image (background)
    base_path = os.path.join(sample_dir, 'base_image.png')
    if os.path.exists(base_path):
        metadata['base_image'] = Image.open(base_path).convert('RGBA')
    
    # Load layer images
    metadata['layer_images'] = {}
    for layer in metadata.get('layers', []):
        img_path = os.path.join(sample_dir, layer['image_path'])
        if os.path.exists(img_path):
            # Load with transparency preserved
            metadata['layer_images'][layer['layer_idx']] = Image.open(img_path).convert('RGBA')
    
    return metadata


def resize_and_crop_to_square(img: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image to fill target_size x target_size, then center crop.
    """
    w, h = img.size
    scale = max(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Center crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def scale_box_xyxy(box: List[int], source_size: int, target_size: int) -> List[int]:
    """
    Scale a box from source_size to target_size.
    Box is in xyxy format: [x0, y0, x1, y1].
    """
    scale = target_size / source_size
    x0, y0, x1, y1 = box
    return [
        int(x0 * scale),
        int(y0 * scale),
        int(x1 * scale),
        int(y1 * scale)
    ]


def compute_random_box_xyxy(
    canvas_size: int,
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.4,
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
) -> List[int]:
    """
    Compute a random bounding box in xyxy format [x0, y0, x1, y1].
    Returns larger boxes to ensure visibility.
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
    
    # Random position
    max_x = max(0, canvas_size - w)
    max_y = max(0, canvas_size - h)
    x0 = random.randint(0, max_x)
    y0 = random.randint(0, max_y)
    x1 = x0 + w
    y1 = y0 + h
    
    return [x0, y0, x1, y1]


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


def compute_smart_box_xyxy(
    canvas_size: int,
    existing_boxes: List[List[int]],
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.4,
    max_attempts: int = 200,
    prefer_corners: bool = True
) -> List[int]:
    """
    Compute a box (xyxy) with minimal overlap with existing boxes.
    
    Strategy:
    1. First try to find a completely non-overlapping position
    2. If not found, find the position with minimum overlap
    3. Prefer corners and edges of the canvas for better placement
    """
    best_box = None
    best_overlap = float('inf')
    
    # Generate candidate boxes
    candidates = []
    
    # Try random positions
    for _ in range(max_attempts):
        box = compute_random_box_xyxy(canvas_size, min_size_ratio, max_size_ratio)
        candidates.append(box)
    
    # Also try corner and edge positions
    if prefer_corners:
        # Get a sample box size
        sample_box = compute_random_box_xyxy(canvas_size, min_size_ratio, max_size_ratio)
        w = sample_box[2] - sample_box[0]
        h = sample_box[3] - sample_box[1]
        
        # Corner positions
        corners = [
            [0, 0, w, h],  # top-left
            [canvas_size - w, 0, canvas_size, h],  # top-right
            [0, canvas_size - h, w, canvas_size],  # bottom-left
            [canvas_size - w, canvas_size - h, canvas_size, canvas_size],  # bottom-right
        ]
        
        # Edge center positions
        edges = [
            [(canvas_size - w) // 2, 0, (canvas_size + w) // 2, h],  # top center
            [(canvas_size - w) // 2, canvas_size - h, (canvas_size + w) // 2, canvas_size],  # bottom center
            [0, (canvas_size - h) // 2, w, (canvas_size + h) // 2],  # left center
            [canvas_size - w, (canvas_size - h) // 2, canvas_size, (canvas_size + h) // 2],  # right center
        ]
        
        candidates.extend(corners)
        candidates.extend(edges)
    
    # Find the best box (minimum overlap)
    for box in candidates:
        # Ensure box is within canvas bounds
        x0, y0, x1, y1 = box
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(canvas_size, x1)
        y1 = min(canvas_size, y1)
        box = [x0, y0, x1, y1]
        
        # Skip invalid boxes
        if x1 <= x0 or y1 <= y0:
            continue
        
        overlap = compute_total_overlap(box, existing_boxes)
        
        # If no overlap, return immediately
        if overlap == 0:
            return box
        
        # Track best box
        if overlap < best_overlap:
            best_overlap = overlap
            best_box = box
    
    # Return the best box found (with minimum overlap)
    if best_box is not None:
        return best_box
    
    # Final fallback
    return compute_random_box_xyxy(canvas_size, min_size_ratio, max_size_ratio)


def compute_non_overlapping_box_xyxy(
    canvas_size: int,
    existing_boxes: List[List[int]],
    min_size_ratio: float = 0.15,
    max_size_ratio: float = 0.4,
    max_attempts: int = 100
) -> List[int]:
    """
    Compute a box (xyxy) with minimal overlap with existing boxes.
    Uses smart placement to find the best position.
    """
    return compute_smart_box_xyxy(
        canvas_size, existing_boxes,
        min_size_ratio, max_size_ratio,
        max_attempts=max_attempts,
        prefer_corners=True
    )


def boxes_overlap_xyxy(box1: List[int], box2: List[int]) -> bool:
    """
    Check if two boxes (xyxy format) overlap.
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # Check for no overlap
    if x0_1 >= x1_2 or x0_2 >= x1_1:
        return False
    if y0_1 >= y1_2 or y0_2 >= y1_1:
        return False
    
    return True


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


def build_whole_caption(
    prism_caption: str,
    laion_caption: Optional[str] = None,
    caption_text: Optional[str] = None
) -> str:
    """
    Build the whole caption by combining prism caption with laion and caption text.
    
    Format: "{prism_caption} With {laion_caption} and Text: {caption_text}."
    
    Example:
        "The image displays a collection of 3D shapes... With Three Dogs Enjoy a 
        Radio Broadcast by Marjorie Turner and Text: catering shop potbelly."
    """
    result = prism_caption.rstrip('.')
    
    additions = []
    if laion_caption:
        additions.append(laion_caption)
    if caption_text:
        additions.append(f"Text: {caption_text}")
    
    if additions:
        result += " With " + " and ".join(additions) + "."
    else:
        result += "."
    
    return result


def xywh_to_xyxy(box: Tuple[int, int, int, int]) -> List[int]:
    """Convert (x, y, w, h) to [x0, y0, x1, y1]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: List[int]) -> Tuple[int, int, int, int]:
    """Convert [x0, y0, x1, y1] to (x, y, w, h)."""
    x0, y0, x1, y1 = box
    return (x0, y0, x1 - x0, y1 - y0)


def get_box_size(box: List[int]) -> Tuple[int, int]:
    """Get width and height from xyxy box."""
    x0, y0, x1, y1 = box
    return (x1 - x0, y1 - y0)


def quantize_box_16(box: List[int], image_size: int) -> List[int]:
    """
    Quantize box to 16-pixel grid (for latent space alignment).
    Input box is in xyxy format.
    """
    x0, y0, x1, y1 = box
    
    # Quantize to 16-pixel grid
    x0_q = (x0 // 16) * 16
    y0_q = (y0 // 16) * 16
    x1_q = ((x1 + 15) // 16) * 16
    y1_q = ((y1 + 15) // 16) * 16
    
    # Clamp to image bounds
    x0_q = max(0, x0_q)
    y0_q = max(0, y0_q)
    x1_q = min(image_size, x1_q)
    y1_q = min(image_size, y1_q)
    
    return [x0_q, y0_q, x1_q, y1_q]


def get_prism_sample_dirs(prism_dir: str, max_samples: Optional[int] = None) -> List[str]:
    """
    Get list of sample directories in PrismLayersPro directory.
    """
    sample_dirs = []
    for name in sorted(os.listdir(prism_dir)):
        if name.startswith('sample_'):
            sample_dirs.append(os.path.join(prism_dir, name))
            if max_samples and len(sample_dirs) >= max_samples:
                break
    return sample_dirs


def load_caption_list(caption_jsonl: str) -> List[Dict]:
    """
    Load captions.jsonl as a list (ordered by line number).
    Each entry has: path, caption, x, y, box_w, box_h, width, height
    """
    return load_jsonl(caption_jsonl)


def get_laion_caption_from_json(image_path: str) -> str:
    """
    Get LAION image caption from its corresponding .json file.
    
    Example:
        /project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image/00000/000000000.jpg
        -> /project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image/00000/000000000.json
        -> {"caption": "wedding day with dog vector image", ...}
    """
    # Replace image extension with .json
    json_path = image_path.rsplit('.', 1)[0] + '.json'
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('caption', '')
        except Exception:
            pass
    
    # Fallback: use filename
    return os.path.basename(image_path).rsplit('.', 1)[0]


def get_laion_images_with_captions(laion_dir: str, laion_jsonl: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get all LAION images with their captions.
    Caption is read from the .json file next to each image.
    Returns list of (image_path, caption) tuples.
    """
    images = []
    
    # Walk through subdirectories
    for subdir in sorted(os.listdir(laion_dir)):
        subdir_path = os.path.join(laion_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in sorted(os.listdir(subdir_path)):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(subdir_path, fname)
                    # Get caption from corresponding .json file
                    caption = get_laion_caption_from_json(img_path)
                    images.append((img_path, caption))
    
    return images


def get_caption_images_with_text(caption_dir: str, caption_list: List[Dict]) -> List[Tuple[str, str]]:
    """
    Get caption images with their text content.
    
    The caption image filename index (e.g., 0000002.png -> index 2)
    corresponds to the line number in captions.jsonl (0-indexed).
    
    Returns list of (image_path, caption_text) tuples.
    """
    images = []
    
    for fname in sorted(os.listdir(caption_dir)):
        if fname.endswith('.png'):
            img_path = os.path.join(caption_dir, fname)
            
            # Extract index from filename (e.g., "0000002.png" -> 2)
            idx_str = fname.split('.')[0]
            try:
                idx = int(idx_str)
            except ValueError:
                idx = -1
            
            # Get caption from the list by index
            caption_text = ""
            if 0 <= idx < len(caption_list):
                caption_text = caption_list[idx].get('caption', '')
            
            images.append((img_path, caption_text))
    
    return images
