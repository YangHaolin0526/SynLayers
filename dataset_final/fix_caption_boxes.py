import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


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


def fix_sample(sample_dir):
    """
    Fix caption-layer bboxes in a single sample directory.
    Returns number of caption layers fixed.
    """
    fixed = 0

    for meta_fname in ['metadata.json', 'metadata_old.json']:
        meta_path = os.path.join(sample_dir, meta_fname)
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        changed = False
        for layer in meta.get('layers', []):
            if layer.get('type') != 'caption':
                continue

            img_path = os.path.join(sample_dir, layer.get('image_path', ''))
            if not os.path.exists(img_path):
                continue

            old_box = layer.get('box', [])
            tight_box = get_content_bbox(img_path)
            if tight_box is not None and tight_box != old_box:
                layer['box'] = tight_box
                layer['width_dst'] = tight_box[2] - tight_box[0]
                layer['height_dst'] = tight_box[3] - tight_box[1]
                changed = True
                fixed += 1

        if changed:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    return fixed


def fix_sample_wrapper(sample_dir):
    """Wrapper for multiprocessing (catches exceptions)."""
    try:
        return fix_sample(sample_dir)
    except Exception as e:
        print(f"Error processing {sample_dir}: {e}")
        return 0


def fix_jsonl(jsonl_path, dataset_dir):
    """
    Fix caption-layer bboxes in a JSONL index file by reading the actual
    layer images from disk.
    Returns number of caption layers fixed.
    """
    entries = []
    fixed = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sample_dir_name = entry.get('sample_dir', '')

            for layer in entry.get('layers', []):
                if layer.get('type') != 'caption':
                    continue
                img_path = os.path.join(dataset_dir, sample_dir_name,
                                        layer.get('image_path', ''))
                if not os.path.exists(img_path):
                    continue

                old_box = layer.get('box', [])
                tight_box = get_content_bbox(img_path)
                if tight_box is not None and tight_box != old_box:
                    layer['box'] = tight_box
                    layer['width_dst'] = tight_box[2] - tight_box[0]
                    layer['height_dst'] = tight_box[3] - tight_box[1]
                    fixed += 1

            entries.append(entry)

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return fixed


def main():
    parser = argparse.ArgumentParser(
        description='Fix caption-layer bounding boxes in a dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers for sample processing')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Find all sample directories
    sample_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if d.startswith('sample_') and os.path.isdir(os.path.join(dataset_dir, d))
    ])
    print(f"Dataset: {dataset_dir}")
    print(f"Samples: {len(sample_dirs)}")
    print(f"Workers: {args.num_workers}")

    # Step 1: Fix metadata.json in each sample directory
    print(f"\n[Step 1/2] Fixing metadata.json files...")
    total_fixed = 0

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            for fixed in tqdm(
                pool.imap_unordered(fix_sample_wrapper, sample_dirs),
                total=len(sample_dirs),
                desc="Fixing samples"
            ):
                total_fixed += fixed
    else:
        for sample_dir in tqdm(sample_dirs, desc="Fixing samples"):
            total_fixed += fix_sample(sample_dir)

    print(f"  Fixed {total_fixed} caption layers in metadata.json files")

    # Step 2: Fix JSONL index files
    print(f"\n[Step 2/2] Fixing JSONL index files...")
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith('.jsonl'):
            continue
        jsonl_path = os.path.join(dataset_dir, fname)
        print(f"  Processing {fname}...")
        jsonl_fixed = fix_jsonl(jsonl_path, dataset_dir)
        print(f"    Fixed {jsonl_fixed} caption layers")

    print(f"\nDone! Total caption layers fixed in metadata: {total_fixed}")


if __name__ == '__main__':
    main()

