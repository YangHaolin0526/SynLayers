import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


SRC_SIZE = 1024
DST_SIZE = 512
SCALE = DST_SIZE / SRC_SIZE  # 0.5


def scale_box(box, scale):
    """Scale xyxy box."""
    return [int(v * scale) for v in box]


def process_sample(task):
    """Process a single sample: downscale images and update metadata."""
    src_dir, dst_dir = task

    os.makedirs(dst_dir, exist_ok=True)

    # --- Resize full-canvas images (whole_image, base_image) ---
    for fname in ['whole_image.png', 'base_image.png']:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.exists(src_path):
            img = Image.open(src_path)
            img = img.resize((DST_SIZE, DST_SIZE), Image.LANCZOS)
            img.save(dst_path)

    # --- Load metadata ---
    meta_path = os.path.join(src_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # --- Resize each layer (cropped images) and scale boxes ---
    for layer in meta.get('layers', []):
        img_name = layer.get('image_path', '')
        src_layer = os.path.join(src_dir, img_name)
        dst_layer = os.path.join(dst_dir, img_name)

        if os.path.exists(src_layer):
            img = Image.open(src_layer)
            # Cropped layer: scale proportionally
            new_w = max(1, int(img.width * SCALE))
            new_h = max(1, int(img.height * SCALE))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(dst_layer)

        # Scale bounding box
        if 'box' in layer:
            layer['box'] = scale_box(layer['box'], SCALE)
        if 'width_dst' in layer:
            layer['width_dst'] = int(layer['width_dst'] * SCALE)
        if 'height_dst' in layer:
            layer['height_dst'] = int(layer['height_dst'] * SCALE)

    # --- Update metadata ---
    meta['width'] = DST_SIZE
    meta['height'] = DST_SIZE
    # Ensure sample_dir is set (needed by PrismBlendDataset)
    sample_name = os.path.basename(dst_dir)
    meta['sample_dir'] = sample_name

    with open(os.path.join(dst_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main():
    parser = argparse.ArgumentParser(description='Downscale PrismLayersPro-image to 512x512')
    parser.add_argument('--src', type=str, required=True,
                        help='Source directory (1024x1024)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination directory (512x512)')
    parser.add_argument('--jsonl', type=str, default=None,
                        help='Source JSONL to filter samples (if not set, process all sample_ dirs)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Determine which samples to process
    if args.jsonl and os.path.exists(args.jsonl):
        sample_names = []
        with open(args.jsonl, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    sample_names.append(d['sample_dir'])
        print(f"Loaded {len(sample_names)} samples from JSONL")
    else:
        sample_names = sorted([
            d for d in os.listdir(args.src)
            if d.startswith('sample_') and os.path.isdir(os.path.join(args.src, d))
        ])
        print(f"Found {len(sample_names)} sample directories")

    # Build task list
    tasks = [
        (os.path.join(args.src, name), os.path.join(args.dst, name))
        for name in sample_names
    ]

    # Process with multiprocessing
    all_metadata = []

    if args.num_workers > 0:
        print(f"Processing with {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            for meta in tqdm(pool.imap_unordered(process_sample, tasks), total=len(tasks)):
                if meta:
                    all_metadata.append(meta)
    else:
        print("Processing single-threaded...")
        for task in tqdm(tasks):
            meta = process_sample(task)
            if meta:
                all_metadata.append(meta)

    # Sort by ID for deterministic output
    all_metadata.sort(key=lambda m: m.get('id', ''))

    # Save JSONL index
    jsonl_out = os.path.join(args.dst, 'train_meta.jsonl')
    with open(jsonl_out, 'w', encoding='utf-8') as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    # Also save index.jsonl (full index)
    index_out = os.path.join(args.dst, 'index.jsonl')
    with open(index_out, 'w', encoding='utf-8') as f:
        for meta in all_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    print(f"\nDone! Downscaled {len(all_metadata)} samples")
    print(f"  Output: {args.dst}")
    print(f"  JSONL:  {jsonl_out}")
    print(f"  Size:   {SRC_SIZE}x{SRC_SIZE} -> {DST_SIZE}x{DST_SIZE}")


if __name__ == '__main__':
    main()

