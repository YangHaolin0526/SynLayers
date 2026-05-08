import os
import json
import shutil
import argparse
import logging
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

CANVAS_SIZE = 1024


def convert_sample(sample_dir, output_sample_dir):
    meta_path = os.path.join(sample_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        logger.warning(f"No metadata.json in {sample_dir}, skipping")
        return False

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    os.makedirs(output_sample_dir, exist_ok=True)

    # Copy non-layer files as-is (whole_image.png, base_image.png, metadata.json)
    for fname in ['whole_image.png', 'base_image.png', 'metadata.json']:
        src = os.path.join(sample_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_sample_dir, fname))

    # Convert each layer
    for layer_info in meta.get('layers', []):
        img_path = os.path.join(sample_dir, layer_info['image_path'])
        if not os.path.exists(img_path):
            logger.warning(f"Layer file not found: {img_path}")
            continue

        box = layer_info['box']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            logger.warning(f"Invalid box {box} for {img_path}")
            continue

        layer_img = Image.open(img_path).convert("RGBA")
        layer_resized = layer_img.resize((box_w, box_h), Image.LANCZOS)

        canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        canvas.paste(layer_resized, (x1, y1))

        out_path = os.path.join(output_sample_dir, layer_info['image_path'])
        canvas.save(out_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PrismLayersPro-testset-1024-full layers to full-canvas format"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to PrismLayersPro-testset-1024-full")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted dataset")
    args = parser.parse_args()

    samples = sorted([
        d for d in os.listdir(args.input_dir)
        if d.startswith('sample_') and os.path.isdir(os.path.join(args.input_dir, d))
    ])
    logger.info(f"Found {len(samples)} samples to convert")

    converted = 0
    for sample_name in tqdm(samples, desc="Converting samples"):
        sample_dir = os.path.join(args.input_dir, sample_name)
        output_sample_dir = os.path.join(args.output_dir, sample_name)
        if convert_sample(sample_dir, output_sample_dir):
            converted += 1

    logger.info(f"Conversion complete: {converted}/{len(samples)} samples converted")
    logger.info(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
