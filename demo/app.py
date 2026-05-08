"""
Local Gradio demo for SynLayers.

Launch with:
    python demo/app.py

Then open http://localhost:7860 in your browser.

Before running, make sure checkpoints are downloaded:
    python tools/download_ckpt.py --download-synlayers
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer.real_infer import run_single_image  # noqa: E402

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "infer" / "infer.yaml"
DEFAULT_BBOX_MODEL = os.environ.get("SYNLAYERS_BBOX_MODEL", "SynLayers/Bbox-caption-8b")
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "demo"


def build_gallery(result: dict) -> list[tuple[str, str]]:
    gallery: list[tuple[str, str]] = []
    if result.get("whole_image_rgba"):
        gallery.append((result["whole_image_rgba"], "Whole RGBA"))
    if result.get("background_rgba"):
        gallery.append((result["background_rgba"], "Background RGBA"))
    for idx, path in enumerate(result.get("layer_images", [])):
        gallery.append((path, f"Layer {idx}"))
    return gallery


def run_demo(image_path: str, seed_value: float, max_new_tokens: int):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU is required.")

    seed = int(seed_value) if seed_value >= 0 else None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = DEFAULT_WORK_DIR / timestamp

    try:
        result = run_single_image(
            image_path=image_path,
            work_dir=str(work_dir),
            bbox_model=DEFAULT_BBOX_MODEL,
            config_path=str(DEFAULT_CONFIG_PATH),
            seed=seed,
            max_new_tokens=int(max_new_tokens),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    return (
        result.get("bbox_visualization", ""),
        result.get("merged_image", ""),
        result.get("caption", ""),
        result.get("metadata", {}),
        build_gallery(result),
        result.get("case_dir", ""),
    )


with gr.Blocks(title="SynLayers Demo") as demo:
    gr.Markdown(
        """
        # SynLayers — Real-World Image Decomposition

        Upload an image to run the full pipeline:
        1. **Stage 1**: Qwen3-VL generates a caption and bounding boxes
        2. **Stage 2**: FLUX-based model decomposes the image into transparent RGBA layers

        **Requires**: NVIDIA GPU ≥ 24 GB VRAM
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Image")
            seed_input = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
            max_new_tokens_input = gr.Slider(
                minimum=128, maximum=2048, value=1024, step=64,
                label="VLM Max New Tokens"
            )
            run_button = gr.Button("Run Pipeline", variant="primary")

        with gr.Column(scale=1):
            bbox_vis_output = gr.Image(type="filepath", label="Detected Bounding Boxes")
            merged_output = gr.Image(type="filepath", label="Merged Decomposition")
            caption_output = gr.Textbox(label="Caption", lines=4)

    with gr.Row():
        meta_json_output = gr.JSON(label="Inference Metadata")
        case_dir_output = gr.Textbox(label="Output Directory")

    layer_gallery = gr.Gallery(label="Predicted Layers", columns=4, height="auto")

    run_button.click(
        fn=run_demo,
        inputs=[image_input, seed_input, max_new_tokens_input],
        outputs=[
            bbox_vis_output,
            merged_output,
            caption_output,
            meta_json_output,
            layer_gallery,
            case_dir_output,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )
