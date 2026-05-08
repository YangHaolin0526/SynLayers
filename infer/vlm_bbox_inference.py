#!/usr/bin/env python3
"""
Shared utility module for VLM bbox-only inference.

This module provides:
- model and processor loading
- prompts for two coordinate conventions
- parsing utilities for bbox-only outputs

The model output is expected to be either:
    [[x0, y0, x1, y1], ...]
or:
    [[x_left, y_top, x_right, y_bottom], ...]

No caption is generated in this module.
"""

import ast
import re
from pathlib import Path

import torch
from transformers import AutoProcessor


# Bottom-left origin, y-axis upward.
BBOX_PROMPT_BOTTOM_LEFT = (
    "<image>This image is 1024 pixels in width and 1024 pixels in height. "
    "The coordinate origin is at the bottom-left corner: x increases to the right, y increases upward. "
    "Detect all objects (layers) in this image. "
    "Output bounding boxes as a list of [x0, y0, x1, y1] where x0=left, y0=bottom, x1=right, y1=top (pixel coordinates). "
    "Output only the list, e.g. [[x0,y0,x1,y1], ...], no other text."
)

# Top-left origin, y-axis downward.
BBOX_PROMPT_TOP_LEFT = (
    "<image>This image is 1024 pixels in width and 1024 pixels in height. "
    "The coordinate origin is at the top-left corner of the image: x increases to the right, y increases downward. "
    "Detect all objects (layers) in this image. "
    "Output bounding boxes as a list of [x_left, y_top, x_right, y_bottom] in pixel coordinates (top-left origin, y downward). "
    "Output only the list, e.g. [[x_left,y_top,x_right,y_bottom], ...], no other text."
)

# Default prompt used by the generic inference API.
BBOX_PROMPT = BBOX_PROMPT_TOP_LEFT
BBOX_SYSTEM = None


def get_model_and_processor(model_path: str, device_map: str = "auto"):
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModel
            model_cls = AutoModel

    base_2b = "Qwen/Qwen3-VL-2B-Instruct"
    base_8b = "Qwen/Qwen3-VL-8B-Instruct"
    base_name = base_8b if "8b" in model_path.lower() or "8B" in model_path else base_2b

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            config.rope_scaling = {}
    except Exception:
        config = None

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_dir = Path(model_path)

    if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
        from peft import PeftModel

        model = model_cls.from_pretrained(
            base_name,
            config=config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(model_dir))

        try:
            processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_name, trust_remote_code=True)
    else:
        model = model_cls.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(base_name, trust_remote_code=True)

    return model, processor


def parse_bbox_output(text: str):
    """
    Parse bbox lists from model output.

    The parser extracts all standalone [a, b, c, d] patterns using regex,
    ignoring outer brackets and extra text. Duplicate boxes are removed to
    reduce the impact of repeated model outputs.
    """
    text = (text or "").strip()

    # Match standalone boxes such as [102, 611, 511, 1023].
    # Both integers and floating-point numbers are supported.
    pattern = (
        r"\[\s*-?\d+(?:\.\d+)?\s*,"
        r"\s*-?\d+(?:\.\d+)?\s*,"
        r"\s*-?\d+(?:\.\d+)?\s*,"
        r"\s*-?\d+(?:\.\d+)?\s*\]"
    )
    matches = re.findall(pattern, text)

    parsed_boxes = []
    for match_str in matches:
        try:
            box = ast.literal_eval(match_str)
            if isinstance(box, list) and len(box) == 4:
                parsed_boxes.append(box)
        except (ValueError, SyntaxError):
            continue

    # Remove duplicate boxes to avoid repeated outputs.
    unique_boxes = []
    seen = set()

    for b in parsed_boxes:
        key = tuple(float(x) for x in b)
        if key not in seen:
            seen.add(key)
            unique_boxes.append(b)

    return unique_boxes


def infer_bboxes(image_path: str, model, processor, *, prompt: str, max_new_tokens: int = 512):
    """
    Run bbox inference for a single image.

    Returns:
        A list of boxes, where each box is [a, b, c, d].
        The coordinate meaning depends on the prompt convention.
    """
    path = Path(image_path)
    if not path.exists():
        return []

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
        clean_up_tokenization_spaces=True,
    )

    raw = (output_text[0] or "").strip()
    boxes = parse_bbox_output(raw)

    result = []
    for b in boxes:
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            result.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])

    return result


def detect_objects(image_path: str, model, processor, *, prompt=None, system=None, max_new_tokens=512):
    """
    Compatibility wrapper using the default bbox prompt.

    The `system` argument is kept for compatibility with older call sites.
    """
    return infer_bboxes(
        image_path,
        model,
        processor,
        prompt=prompt or BBOX_PROMPT,
        max_new_tokens=max_new_tokens,
    )
