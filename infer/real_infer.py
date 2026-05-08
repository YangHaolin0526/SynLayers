import os
import re
import sys
import json
import logging
import argparse

# Add project root to Python path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress CLIP tokenizer truncation warnings (CLIP truncates internally, T5 handles full caption)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image

from infer.prism_infer import initialize_pipeline, scale_box_xyxy, quantize_box_16
from tools.tools import load_config, seed_everything


def load_real_metadata(jsonl_path: str):
    """Load real-test metadata from JSONL."""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_checkpoint_tag(path: str):
    """Extract a checkpoint tag like scaleup_1024_20k or original_1024_512seq."""
    if not path:
        return None

    match = re.search(r"ckpt_prism_([^/]+)", path)
    if match:
        return match.group(1)
    return None


def derive_run_name(config: dict) -> str:
    """Derive the result subfolder name from the active checkpoint setup."""
    checkpoint_tags = {}
    for key in ("lora_ckpt", "layer_ckpt", "adapter_lora_dir"):
        tag = extract_checkpoint_tag(config.get(key, ""))
        if tag:
            checkpoint_tags[key] = tag

    if checkpoint_tags:
        unique_tags = sorted(set(checkpoint_tags.values()))
        if len(unique_tags) != 1:
            details = ", ".join(f"{key}={value}" for key, value in checkpoint_tags.items())
            raise ValueError(
                "Checkpoint paths are inconsistent. "
                "Please switch lora_ckpt, layer_ckpt, and adapter_lora_dir together. "
                f"Current tags: {details}"
            )
        inferred_tag = unique_tags[0]
    else:
        inferred_tag = "real_infer"

    if config.get("run_name"):
        return config["run_name"]
    return inferred_tag


def build_run_save_dir(config: dict):
    """Build the final save directory as <save_dir>/<run_name>."""
    save_root = config.get("save_dir", "./real_inference_output")
    run_name = derive_run_name(config)
    return os.path.join(save_root, run_name), run_name


def resolve_image_path(sample: dict, data_dir: str, image_dir: str = None) -> str:
    """Resolve the input image path, preferring local files_real_test images."""
    sample_name = sample.get("sample_or_stem", "")
    image_path = sample.get("image", "")

    if image_dir is None and data_dir:
        image_dir = os.path.join(data_dir, "layers_real_test_1024")

    candidates = []

    if image_dir:
        if sample_name:
            candidates.extend([
                os.path.join(image_dir, f"{sample_name}.png"),
                os.path.join(image_dir, f"{sample_name}.jpg"),
                os.path.join(image_dir, f"{sample_name}.jpeg"),
            ])
        if image_path:
            candidates.append(os.path.join(image_dir, os.path.basename(image_path)))

    if image_path:
        candidates.append(image_path)
        if data_dir and not os.path.isabs(image_path):
            candidates.append(os.path.join(data_dir, image_path))

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image for sample '{sample_name}'. "
        f"Tried local image_dir='{image_dir}' and json path '{image_path}'."
    )


def quantize_box_16_safe(box: tuple, target_size: int) -> tuple:
    """Quantize a box to the 16-pixel grid and keep at least one latent cell."""
    x0_q, y0_q, x1_q, y1_q = quantize_box_16(box, target_size)

    if x1_q <= x0_q:
        if x0_q + 16 <= target_size:
            x1_q = x0_q + 16
        else:
            x0_q = max(0, target_size - 16)
            x1_q = target_size

    if y1_q <= y0_q:
        if y0_q + 16 <= target_size:
            y1_q = y0_q + 16
        else:
            y0_q = max(0, target_size - 16)
            y1_q = target_size

    return (x0_q, y0_q, x1_q, y1_q)


def get_real_boxes(sample: dict, source_size: int, target_size: int) -> list:
    """Scale and quantize real-test boxes from JSON metadata."""
    boxes = []
    for box in sample.get("bboxes", []):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        scaled_box = scale_box_xyxy(box, source_size, target_size)
        boxes.append(quantize_box_16_safe(scaled_box, target_size))
    return boxes


def load_adapter_image(sample: dict, target_size: int, config: dict):
    """Load and resize the real-test image used as adapter input."""
    image_path = resolve_image_path(
        sample,
        data_dir=config.get("data_dir", ""),
        image_dir=config.get("image_dir"),
    )
    img = Image.open(image_path).convert("RGB")

    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)

    return img, image_path


@torch.no_grad()
def inference_real(config):
    """Main inference function for the real-test dataset."""
    if config.get("seed") is not None:
        seed_everything(config["seed"])

    source_size = config.get("source_size", 1024)
    target_size = config.get("target_size", 1024)
    max_layer_num = config.get("max_layer_num", 52)

    print(f"[INFO] Source size: {source_size}, Target size: {target_size}", flush=True)

    save_dir, run_name = build_run_save_dir(config)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "merged"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "merged_rgba"), exist_ok=True)
    print(f"[INFO] Run name: {run_name}", flush=True)
    print(f"[INFO] Results will be saved to: {save_dir}", flush=True)

    pipeline, transp_vae = initialize_pipeline(config)

    test_jsonl = config.get("test_jsonl", "")
    if not test_jsonl or not os.path.exists(test_jsonl):
        raise ValueError(f"Test JSONL not found: {test_jsonl}")

    all_samples = load_real_metadata(test_jsonl)
    total_available = len(all_samples)

    start_idx = config.get("start_idx", 1)
    end_idx = config.get("end_idx", total_available)
    max_samples = config.get("max_samples", None)

    if max_samples and not config.get("end_idx"):
        end_idx = min(start_idx + max_samples - 1, total_available)

    start_idx = max(1, min(start_idx, total_available))
    end_idx = max(start_idx, min(end_idx, total_available))
    samples = all_samples[start_idx - 1 : end_idx]

    print(f"[INFO] Total samples in dataset: {total_available}", flush=True)
    print(f"[INFO] Processing samples {start_idx} to {end_idx} ({len(samples)} samples)", flush=True)

    generator = torch.Generator(device=torch.device("cuda")).manual_seed(config.get("seed", 42))

    for local_idx, sample in enumerate(samples):
        idx_zero_based = start_idx - 1 + local_idx
        sample_name = sample.get("sample_or_stem", f"real_{idx_zero_based:06d}")
        print(f"Processing [{local_idx + 1}/{len(samples)}] idx={idx_zero_based} ({sample_name})...", flush=True)

        try:
            layer_boxes = get_real_boxes(sample, source_size, target_size)
            adapter_img, image_path = load_adapter_image(sample, target_size, config)
        except Exception as e:
            print(f"  Error preparing sample: {e}", flush=True)
            continue

        whole_box = (0, 0, target_size, target_size)
        bg_box = (0, 0, target_size, target_size)
        all_boxes = [whole_box, bg_box] + layer_boxes

        if len(all_boxes) > max_layer_num:
            print(
                f"  Skipping sample because num_layers={len(all_boxes)} exceeds max_layer_num={max_layer_num}",
                flush=True,
            )
            continue

        caption = sample.get("whole_caption", "")
        print(f"  Size: {target_size}x{target_size}, Layers: {len(all_boxes)}", flush=True)

        try:
            x_hat, image, _ = pipeline(
                prompt=caption,
                adapter_image=adapter_img,
                adapter_conditioning_scale=config.get("adapter_scale", 0.9),
                validation_box=all_boxes,
                generator=generator,
                height=target_size,
                width=target_size,
                guidance_scale=config.get("cfg", 4.0),
                num_layers=len(all_boxes),
                sdxl_vae=transp_vae,
            )
        except Exception as e:
            print(f"  Error during inference: {e}", flush=True)
            continue

        x_hat = (x_hat + 1) / 2
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).to(torch.float32)

        case_dir = os.path.join(save_dir, sample_name)
        os.makedirs(case_dir, exist_ok=True)

        whole_image_layer = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(whole_image_layer, "RGBA").save(os.path.join(case_dir, "whole_image_rgba.png"))

        background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(background_layer, "RGBA").save(os.path.join(case_dir, "background_rgba.png"))

        adapter_img.save(os.path.join(case_dir, "origin.png"))

        merged_image = image[1]
        for layer_idx in range(2, x_hat.shape[0]):
            rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgba_image = Image.fromarray(rgba_layer, "RGBA")
            rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx - 2}_rgba.png"))
            merged_image = Image.alpha_composite(merged_image.convert("RGBA"), rgba_image)

        merged_image.convert("RGB").save(os.path.join(save_dir, "merged", f"{sample_name}.png"))
        merged_image.convert("RGB").save(os.path.join(case_dir, "merged.png"))
        merged_image.save(os.path.join(save_dir, "merged_rgba", f"{sample_name}.png"))

        case_meta = {
            "sample_idx_zero_based": idx_zero_based,
            "sample_idx_one_based": idx_zero_based + 1,
            "sample_name": sample_name,
            "source_image_path": image_path,
            "target_size": target_size,
            "source_size": source_size,
            "raw_num_layers": sample.get("num_layers"),
            "num_layers": len(all_boxes),
            "raw_boxes": sample.get("bboxes", []),
            "boxes": all_boxes,
            "caption": caption,
            "run_name": run_name,
        }
        with open(os.path.join(case_dir, "inference_meta.json"), "w", encoding="utf-8") as f:
            json.dump(case_meta, f, indent=2)

        if idx_zero_based % 10 == 0:
            torch.cuda.empty_cache()

    print(f"[INFO] Inference complete. Results saved to {save_dir}", flush=True)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="1-based start index for the JSONL entries.",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="1-based end index for the JSONL entries (inclusive).",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    if args.start_idx is not None:
        config["start_idx"] = args.start_idx
    if args.end_idx is not None:
        config["end_idx"] = args.end_idx
    inference_real(config)
