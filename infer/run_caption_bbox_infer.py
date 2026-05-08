#!/usr/bin/env python3
"""
Local model inference: generate whole_caption + bounding boxes
(consistent with build_caption_bbox_numlayers_data / caption_bbox_sft format).

- Prompt: origin at top-left, y-axis downward.
  Output JSON format:
    {"whole_caption": "...", "boxes": [[x_left,y_top,x_right,y_bottom], ...]}

- Parse whole_caption and bounding boxes from model output.
- Compute number of boxes as num_layers = len(bboxes).

- Output JSONL per line:
  sample_or_stem, image, whole_caption, bboxes, num_layers, coord

Model: Qwen3-VL-8B bbox-caption LoRA (user-specified path)
"""

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from vlm_bbox_inference import (
    get_model_and_processor,
    parse_bbox_output,
)

# Prompt definition (top-left coordinate system)
CAPTION_BBOX_PROMPT_TOP_LEFT = (
    "<image>This image is 1024 pixels in width and 1024 pixels in height. "
    "The coordinate origin is at the top-left corner of the image: x increases to the right, y increases downward. "
    "First describe the whole image in one detailed caption (whole_caption). "
    "Then list the bounding box for each visible layer or object. "
    "Each box is [x_left, y_top, x_right, y_bottom] in pixel coordinates (top-left origin, y downward). "
    "Output a single JSON object with exactly two keys: \"whole_caption\" (string) and \"boxes\" (list of [x_left,y_top,x_right,y_bottom] arrays). "
    "Output only this JSON, no other text or markdown."
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_json_caption_bbox(text: str):
    """
    Parse JSON output from model: extract whole_caption and bounding boxes.

    If JSON parsing fails or model outputs only bbox list,
    fallback to parse_bbox_output and set caption to empty string.

    Returns:
        (whole_caption: str, bboxes: list)
    """
    text = (text or "").strip()

    # Remove markdown code blocks if present
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                try:
                    obj = json.loads(p)
                    if isinstance(obj, dict):
                        caption = obj.get("whole_caption") or obj.get("caption") or ""
                        boxes = obj.get("boxes") or obj.get("bboxes") or []
                        if isinstance(boxes, list):
                            return caption, boxes
                except json.JSONDecodeError:
                    pass

    # Direct JSON extraction
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                caption = obj.get("whole_caption") or obj.get("caption") or ""
                boxes = obj.get("boxes") or obj.get("bboxes") or []
                if isinstance(boxes, list):
                    return caption, boxes
        except json.JSONDecodeError:
            pass

    # Fallback: bbox-only output
    boxes = parse_bbox_output(text)
    return "", boxes


def collect_images(data_dir: Path, max_samples: int | None, target_samples: set | None = None):
    """
    Collect images from:
    1) sample_*/whole_image.png
    2) flat directory with image files

    Returns list of (name, image_path)
    """
    data_dir = Path(data_dir)
    out = []

    for d in sorted(data_dir.glob("sample_*")):
        if not d.is_dir():
            continue
        if target_samples is not None and d.name not in target_samples:
            continue
        whole = d / "whole_image.png"
        if whole.exists():
            out.append((d.name, str(whole.resolve())))
            if max_samples and len(out) >= max_samples:
                return out

    # Fallback: direct image files
    if not out:
        def _sort_key(p: Path):
            # Sort by trailing number if present
            parts = p.stem.rsplit("_", 1)
            try:
                return (parts[0], int(parts[-1]))
            except ValueError:
                return (p.stem, 0)

        all_imgs = [
            p for ext in IMAGE_EXTS
            for p in data_dir.glob(f"*{ext}")
            if p.is_file()
        ]

        for p in sorted(all_imgs, key=_sort_key):
            if target_samples is not None and p.stem not in target_samples:
                continue
            out.append((p.stem, str(p.resolve())))
            if max_samples and len(out) >= max_samples:
                return out

    return out


def draw_boxes(image_path: Path, bboxes: list, out_path: Path, color: str = "lime", width: int = 3):
    """
    Draw bounding boxes on image.

    Input format: [x_left, y_top, x_right, y_bottom]
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for b in bboxes:
        if len(b) != 4:
            continue
        x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def infer_caption_bbox(image_path: str, model, processor, *, prompt: str, max_new_tokens: int = 1024):
    """
    Run caption + bbox inference for a single image.

    Returns:
        (whole_caption, bboxes)
    """
    path = Path(image_path)
    if not path.exists():
        return "", []

    content = [
        {"type": "image", "image": str(path.absolute())},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    output_ids = generated[:, input_len:]

    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    raw = (output_text[0] or "").strip()
    whole_caption, bboxes = parse_json_caption_bbox(raw)

    # Normalize bbox format
    result_boxes = []
    for b in bboxes:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            result_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    return whole_caption, result_boxes


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Caption + bbox inference (top-left origin)"
    )

    parser.add_argument("--data-dir", type=str, default="testset",
                        help="Directory containing sample_* or image files")
    parser.add_argument("--output", type=str, default="caption_bbox_infer.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (merged or LoRA)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--samples", type=str, nargs="+",
                        help="Specify sample names (e.g. sample_001)")
    parser.add_argument("--vis-dir", type=str, default=None,
                        help="Optional directory for visualization")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    target_samples = set(args.samples) if args.samples else None

    rows = collect_images(data_dir, args.max_samples, target_samples)
    if not rows:
        print(f"No images found under {data_dir}")
        return

    print(f"Loading model: {args.model}")
    model, processor = get_model_and_processor(args.model)

    print(f"Running inference on {len(rows)} samples...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    with open(out_path, "w", encoding="utf-8") as f:
        for name, image_path in rows:
            print(f"  {name}")

            whole_caption, bboxes = infer_caption_bbox(
                image_path,
                model,
                processor,
                prompt=CAPTION_BBOX_PROMPT_TOP_LEFT,
                max_new_tokens=args.max_new_tokens,
            )

            num_layers = len(bboxes)

            record = {
                "sample_or_stem": name,
                "image": image_path,
                "whole_caption": whole_caption,
                "bboxes": bboxes,
                "num_layers": num_layers,
                "coord": "top_left",
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if vis_dir:
                draw_boxes(Path(image_path), bboxes, vis_dir / f"{name}_vis.png")

    print(f"Wrote {out_path}")
    if vis_dir:
        print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()