import os
import json
import argparse
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm


def upscale_image(src_path, dst_path, target_size):
    """Load image, resize to target_size x target_size, save."""
    img = Image.open(src_path)
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    img.save(dst_path)


def scale_box(box, scale):
    """Scale xyxy box by a factor."""
    return [int(v * scale) for v in box]


def get_content_bbox(img_path):
    """
    Get the tight bounding box of non-transparent content in an RGBA image.
    Returns [x0, y0, x1, y1] or None if the image is fully transparent.
    """
    img = Image.open(img_path).convert('RGBA')
    arr = np.array(img)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax + 1), int(rmax + 1)]


def process_sample(src_sample_dir, dst_sample_dir, src_size, dst_size):
    """Process a single sample: upscale images and update metadata."""
    os.makedirs(dst_sample_dir, exist_ok=True)
    scale = dst_size / src_size

    # Upscale all PNG images first
    for fname in os.listdir(src_sample_dir):
        src_path = os.path.join(src_sample_dir, fname)
        dst_path = os.path.join(dst_sample_dir, fname)

        if fname.endswith('.png'):
            upscale_image(src_path, dst_path, dst_size)

    # Now process metadata files — for caption layers, recompute bbox from
    # the actual upscaled image content instead of just scaling by 2x
    for fname in ['metadata.json', 'metadata_old.json']:
        src_path = os.path.join(src_sample_dir, fname)
        dst_path = os.path.join(dst_sample_dir, fname)

        if not os.path.exists(src_path):
            continue

            with open(src_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            meta['width'] = dst_size
            meta['height'] = dst_size

            for layer in meta.get('layers', []):
            layer_type = layer.get('type')

            if layer_type == 'caption':
                # Caption layers: compute tight bbox from actual pixel content
                layer_img_path = os.path.join(dst_sample_dir, layer.get('image_path', ''))
                if os.path.exists(layer_img_path):
                    tight_box = get_content_bbox(layer_img_path)
                    if tight_box is not None:
                        layer['box'] = tight_box
                        layer['width_dst'] = tight_box[2] - tight_box[0]
                        layer['height_dst'] = tight_box[3] - tight_box[1]
                    else:
                        # Fallback: simple 2x scale
                        if 'box' in layer:
                            layer['box'] = scale_box(layer['box'], scale)
                        if 'width_dst' in layer:
                            layer['width_dst'] = int(layer['width_dst'] * scale)
                        if 'height_dst' in layer:
                            layer['height_dst'] = int(layer['height_dst'] * scale)
                else:
                if 'box' in layer:
                    layer['box'] = scale_box(layer['box'], scale)
                if 'width_dst' in layer:
                    layer['width_dst'] = int(layer['width_dst'] * scale)
                if 'height_dst' in layer:
                    layer['height_dst'] = int(layer['height_dst'] * scale)
            else:
                # Non-caption layers: simple 2x scale
                if 'box' in layer:
                    layer['box'] = scale_box(layer['box'], scale)
                if 'width_dst' in layer:
                    layer['width_dst'] = int(layer['width_dst'] * scale)
                if 'height_dst' in layer:
                    layer['height_dst'] = int(layer['height_dst'] * scale)

            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)


def upscale_jsonl(src_jsonl, dst_jsonl, src_size, dst_size, dst_dir):
    """Upscale all entries in a JSONL file (boxes + dimensions).
    For caption layers, recompute bbox from the upscaled image content."""
    scale = dst_size / src_size

    with open(src_jsonl, 'r', encoding='utf-8') as f_in, \
         open(dst_jsonl, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            entry['width'] = dst_size
            entry['height'] = dst_size

            sample_dir = entry.get('sample_dir', '')

            for layer in entry.get('layers', []):
                layer_type = layer.get('type')

                if layer_type == 'caption' and sample_dir:
                    # Caption layers: compute tight bbox from actual pixel content
                    layer_img_path = os.path.join(dst_dir, sample_dir, layer.get('image_path', ''))
                    if os.path.exists(layer_img_path):
                        tight_box = get_content_bbox(layer_img_path)
                        if tight_box is not None:
                            layer['box'] = tight_box
                            layer['width_dst'] = tight_box[2] - tight_box[0]
                            layer['height_dst'] = tight_box[3] - tight_box[1]
                            continue
                    # Fallback
                    if 'box' in layer:
                        layer['box'] = scale_box(layer['box'], scale)
                    if 'width_dst' in layer:
                        layer['width_dst'] = int(layer['width_dst'] * scale)
                    if 'height_dst' in layer:
                        layer['height_dst'] = int(layer['height_dst'] * scale)
                else:
                if 'box' in layer:
                    layer['box'] = scale_box(layer['box'], scale)
                if 'width_dst' in layer:
                    layer['width_dst'] = int(layer['width_dst'] * scale)
                if 'height_dst' in layer:
                    layer['height_dst'] = int(layer['height_dst'] * scale)

            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Upscale testset from 512 to 1024')
    parser.add_argument('--src', type=str, required=True,
                        help='Source testset directory (512x512)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination directory (1024x1024)')
    parser.add_argument('--src_size', type=int, default=512)
    parser.add_argument('--dst_size', type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # Find all sample directories
    sample_dirs = sorted([
        d for d in os.listdir(args.src)
        if d.startswith('sample_') and os.path.isdir(os.path.join(args.src, d))
    ])
    print(f"Found {len(sample_dirs)} samples to upscale")

    # Process each sample
    for sample_name in tqdm(sample_dirs, desc="Upscaling samples"):
        src_sample = os.path.join(args.src, sample_name)
        dst_sample = os.path.join(args.dst, sample_name)
        process_sample(src_sample, dst_sample, args.src_size, args.dst_size)

    # Process JSONL files (after images are upscaled so we can read them)
    for jsonl_name in os.listdir(args.src):
        if jsonl_name.endswith('.jsonl'):
            src_jsonl = os.path.join(args.src, jsonl_name)
            dst_jsonl = os.path.join(args.dst, jsonl_name)
            print(f"Upscaling JSONL: {jsonl_name}")
            upscale_jsonl(src_jsonl, dst_jsonl, args.src_size, args.dst_size, args.dst)

    print(f"\nDone! Upscaled testset saved to {args.dst}")
    print(f"  Samples: {len(sample_dirs)}")
    print(f"  Resolution: {args.src_size}x{args.src_size} -> {args.dst_size}x{args.dst_size}")


if __name__ == '__main__':
    main()
