import os
import sys
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from glob import glob


def load_pipeline(model_path, device="cuda", dtype=torch.bfloat16):
    from diffusers import Flux2Pipeline

    print(f"[INFO] Loading FLUX.2-dev from {model_path}...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipe.to(device)
    print("[INFO] FLUX.2-dev loaded.", flush=True)
    return pipe


def refine_image(pipe, image, prompt, num_inference_steps=28, guidance_scale=4.0,
                 seed=42, caption_upsample_temperature=None):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        image=[image],
        height=image.height,
        width=image.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if caption_upsample_temperature is not None:
        kwargs["caption_upsample_temperature"] = caption_upsample_temperature

    result = pipe(**kwargs)
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-dev post-processing refinement")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="SynLayers inference output directory (contains sample_XXXXXX folders)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save refined outputs")
    parser.add_argument("--model_path", type=str,
                        default="",
                        help="Path to FLUX.2-dev model")
    parser.add_argument("--prompt", type=str,
                        default="Refine this layered design image. Maintain the exact composition, "
                                "layout, text, and all visual elements. Improve sharpness, color "
                                "accuracy, material rendering, and remove any artifacts or noise patterns.",
                        help="Editing prompt for refinement")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--caption_upsample_temperature", type=float, default=None,
                        help="Set to 0.15 for caption upsampling (recommended by BFL)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--refine_layers", action="store_true",
                        help="Also refine individual layer RGBA images (slower)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "merged"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)

    pipe = load_pipeline(args.model_path, device=args.device)

    sample_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(args.input_dir, d))
    ])

    if args.max_samples is not None:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"[INFO] Processing {len(sample_dirs)} samples", flush=True)

    for idx, sample_name in enumerate(tqdm(sample_dirs, desc="Refining")):
        src_dir = os.path.join(args.input_dir, sample_name)
        dst_dir = os.path.join(args.output_dir, sample_name)
        os.makedirs(dst_dir, exist_ok=True)

        merged_path = os.path.join(src_dir, "merged.png")
        if not os.path.exists(merged_path):
            print(f"  Skipping {sample_name}: no merged.png", flush=True)
            continue

        original = Image.open(merged_path).convert("RGB")

        refined = refine_image(
            pipe, original, args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            caption_upsample_temperature=args.caption_upsample_temperature,
        )

        refined.save(os.path.join(dst_dir, "merged_refined.png"))
        refined.save(os.path.join(args.output_dir, "merged", f"{sample_name}.png"))

        original.save(os.path.join(dst_dir, "merged_original.png"))

        comparison = Image.new("RGB", (original.width * 2, original.height))
        comparison.paste(original, (0, 0))
        comparison.paste(refined, (original.width, 0))
        comparison.save(os.path.join(args.output_dir, "comparison", f"{sample_name}.png"))

        # Copy over other files from source
        for fname in ["origin.png", "whole_image_rgba.png", "background_rgba.png", "inference_meta.json"]:
            src_file = os.path.join(src_dir, fname)
            if os.path.exists(src_file):
                import shutil
                shutil.copy2(src_file, os.path.join(dst_dir, fname))

        # Copy layer files
        for layer_file in sorted(glob(os.path.join(src_dir, "layer_*_rgba.png"))):
            import shutil
            shutil.copy2(layer_file, os.path.join(dst_dir, os.path.basename(layer_file)))

        if idx % 5 == 0:
            torch.cuda.empty_cache()

    print(f"[INFO] Refinement complete. Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
