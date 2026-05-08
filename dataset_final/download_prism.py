import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# All style splits in PrismLayersPro
STYLE_SPLITS = [
    "3D", "Pokemon", "anime", "cartoon", "doodle_art", "furry", "ink",
    "kid_crayon_drawing", "line_draw", "melting_gold", "melting_silver",
    "metal_textured", "neon_graffiti", "papercut_art", "pixel_art",
    "pop_art", "sand_painting", "steampunk", "toy", "watercolor_painting",
    "wood_carving"
]


def save_image(img, path):
    """Save PIL Image to path, converting to RGBA if needed."""
    if img is None:
        return False
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    img.save(path)
    return True


def is_sample_complete(sample_dir, layer_count):
    """Check if a sample folder is complete (has all required files)."""
    if not sample_dir.exists():
        return False
    
    # Check required files
    required_files = ["whole_image.png", "base_image.png", "metadata.json"]
    for f in required_files:
        if not (sample_dir / f).exists():
            return False
    
    # Check all layer files
    for i in range(layer_count):
        if not (sample_dir / f"layer_{i:02d}.png").exists():
            return False
    
    return True


def process_sample(sample, sample_idx, output_dir, skip_existing=True):
    """Process a single sample and save all images + metadata."""
    sample_dir = output_dir / f"sample_{sample_idx:06d}"
    
    layer_count = sample.get("layer_count", 0)
    
    # Skip if already complete
    if skip_existing and is_sample_complete(sample_dir, layer_count):
        # Load existing metadata for index
        meta_path = sample_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save whole image
    whole_img = sample.get("whole_image")
    if whole_img:
        save_image(whole_img, sample_dir / "whole_image.png")

    # Save base image
    base_img = sample.get("base_image")
    if base_img:
        save_image(base_img, sample_dir / "base_image.png")

    # Prepare metadata
    metadata = {
        "id": sample.get("id", f"sample_{sample_idx:06d}"),
        "style_category": sample.get("style_category", ""),
        "whole_caption": sample.get("whole_caption", ""),
        "base_caption": sample.get("base_caption", ""),
        "layer_count": sample.get("layer_count", 0),
        "layers": []
    }

    # Save each layer
    layer_count = sample.get("layer_count", 0)
    for i in range(layer_count):
        layer_key = f"layer_{i:02d}"
        layer_img = sample.get(layer_key)
        layer_caption = sample.get(f"{layer_key}_caption", "")
        layer_box = sample.get(f"{layer_key}_box", [0, 0, 0, 0])
        layer_width_dst = sample.get(f"{layer_key}_width_dst", 0)
        layer_height_dst = sample.get(f"{layer_key}_height_dst", 0)

        # Save layer image
        layer_path = sample_dir / f"{layer_key}.png"
        if layer_img is not None:
            save_image(layer_img, layer_path)

        # Add to metadata
        metadata["layers"].append({
            "layer_idx": i,
            "caption": layer_caption,
            "box": list(layer_box) if layer_box else [0, 0, 0, 0],  # [x, y, w, h]
            "width_dst": layer_width_dst,
            "height_dst": layer_height_dst,
            "image_path": f"{layer_key}.png"
        })

    # Save metadata
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Download PrismLayersPro dataset")
    parser.add_argument("--output_dir", type=str, default="/project/llmsvgen/share/data/kmw_layered_dataset/PrismLayersPro-image/data/haolin/PrismLayersPro-image",
                        help="Output directory for downloaded images")
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Maximum number of samples to download")
    parser.add_argument("--cache_dir", type=str, default="/project/llmsvgen/share/data/kmw_layered_dataset/PrismLayersPro-image",
                        help="Cache directory for HuggingFace datasets")
    parser.add_argument("--styles", type=str, nargs="+", default=None,
                        help="Specific styles to download (default: all)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip existing complete samples (re-download all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_existing = not args.no_skip
    styles = args.styles if args.styles else STYLE_SPLITS
    print(f"Downloading styles: {styles}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Skip existing: {skip_existing}")

    global_idx = 0
    all_samples_meta = []

    for style in styles:
        if global_idx >= args.max_samples:
            break

        print(f"\n=== Loading style: {style} ===")
        try:
            dataset = load_dataset(
                "artplus/PrismLayersPro",
                split=style,
                cache_dir=args.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load style {style}: {e}")
            continue

        print(f"Found {len(dataset)} samples in {style}")

        skipped = 0
        processed = 0
        for sample in tqdm(dataset, desc=f"Processing {style}"):
            if global_idx >= args.max_samples:
                break

            try:
                meta = process_sample(sample, global_idx, output_dir, skip_existing=skip_existing)
                if meta:
                    meta["global_idx"] = global_idx
                    meta["sample_dir"] = f"sample_{global_idx:06d}"
                    all_samples_meta.append(meta)
                    # Check if this was a skip
                    sample_dir = output_dir / f"sample_{global_idx:06d}"
                    if sample_dir.exists() and (sample_dir / "metadata.json").exists():
                        skipped += 1
                    else:
                        processed += 1
                global_idx += 1
            except Exception as e:
                print(f"Error processing sample {global_idx}: {e}")
                global_idx += 1
                continue
        
        print(f"  {style}: processed={processed}, skipped={skipped}")

    # Save global metadata index
    index_path = output_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for meta in all_samples_meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"\n=== Download complete ===")
    print(f"Total samples: {global_idx}")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()

