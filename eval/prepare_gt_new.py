import os
import json
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare GT index for evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of PrismLayersPro-blended-testset")
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to blend_meta.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON index path")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target evaluation size")
    args = parser.parse_args()

    # Load metadata
    samples = []
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Found {len(samples)} samples in JSONL")

    gt_index = []
    skipped = 0

    for sample in tqdm(samples, desc="Preparing GT index"):
        sample_dir_name = sample.get('sample_dir', '')
        sample_path = os.path.join(args.data_dir, sample_dir_name)

        if not os.path.isdir(sample_path):
            skipped += 1
            continue

        # Verify key files exist
        whole_img = os.path.join(sample_path, 'whole_image.png')
        base_img = os.path.join(sample_path, 'base_image.png')
        if not os.path.exists(whole_img):
            skipped += 1
            continue

        # Extract layer info
        layers = sample.get('layers', [])
        layer_boxes = []
        layer_files = []

        for layer in layers:
            box = layer.get('box', [0, 0, args.target_size, args.target_size])
            img_path = layer.get('image_path', '')
            full_path = os.path.join(sample_path, img_path)

            if os.path.exists(full_path):
                layer_boxes.append(box)
                layer_files.append(img_path)

        entry = {
            'sample_name': sample_dir_name,
            'sample_path': sample_path,
            'whole_image': 'whole_image.png',
            'base_image': 'base_image.png',
            'layer_count': len(layer_files),
            'layer_files': layer_files,
            'layer_boxes': layer_boxes,
            'caption': sample.get('whole_caption', ''),
        }
        gt_index.append(entry)

    # Save index
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(gt_index, f, indent=2, ensure_ascii=False)

    print(f"GT index saved to {args.output}")
    print(f"  Total: {len(gt_index)} samples, Skipped: {skipped}")


if __name__ == '__main__':
    main()

