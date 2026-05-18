import os
import json
import time
import argparse
import torch
from tqdm import tqdm

ROOT_DIR = os.environ.get("ROOT_DIR", "Your_Data_Directory")
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
###
SYSTEM_PROMPT = """You are an expert image captioner. 
Your task is to refine and condense a long, redundant 'whole caption' of a layered image.
The original caption is a combination of a background description and multiple foreground layers with their positions and descriptions.

Requirements:
1. Conciseness: Keep the final caption between 100 to 140 words!
2. Natural Flow: Blend the background and layers into a cohesive, professional paragraph. Avoid repetitive phrases like 'you can see' or 'there is'.
3. Output Format: Return ONLY the refined caption string.                                                
4. Accuracy and Vividness: Ensure descriptions precisely match visual elements, using vivid but concise language; handle any layer overlaps or interactions naturally without redundancy. 
5. Make sure we have the first 50 words of the caption to be a overview of the image. And the rest of the caption, should be a detailed description of the image, around 60 to 100 words.
6. If there contains layers that are overlapped by other layers, you should describe the overlapped layers in the caption as well in a concise and proper manner.
7. For english text layer, you should describe the text in the caption in details, what is it in the text layer.
"""


def load_model(device):
    """Load Qwen2.5-VL-3B-Instruct on a specific device."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"  Loading model weights...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device)
    print(f"  Loading processor...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    processor.tokenizer.padding_side = "left"
    model.eval()
    print(f"  Model ready on {device}")
    return model, processor


def refine_caption_batch(model, processor, whole_captions, whole_image_paths, device):
    """Refine a batch of captions using Qwen2.5-VL with whole_image as visual input."""
    from qwen_vl_utils import process_vision_info

    all_texts = []
    all_image_inputs = []

    for caption, img_path in zip(whole_captions, whole_image_paths):
        content = []
        if img_path and os.path.exists(img_path):
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": f"Refine this caption: {caption}"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)

        img_msg = [{"role": "user", "content": content}]
        img_inputs, _ = process_vision_info(img_msg)
        if img_inputs:
            all_image_inputs.extend(img_inputs)

    inputs = processor(
        text=all_texts,
        images=all_image_inputs if all_image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    results = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [r.strip() for r in results]


def process_sample_check(sample_name, skip_existing=False):
    """Check if a sample needs processing. Returns (sample_name, whole_caption, img_path) or None."""
    sample_path = os.path.join(ROOT_DIR, sample_name)
    metadata_path = os.path.join(sample_path, "metadata.json")
    metadata_old_path = os.path.join(sample_path, "metadata_old.json")

    if skip_existing and os.path.exists(metadata_old_path) and os.path.exists(metadata_path):
        return None

    if os.path.exists(metadata_old_path):
        src = metadata_old_path
    elif os.path.exists(metadata_path):
        os.rename(metadata_path, metadata_old_path)
        src = metadata_old_path
    else:
        return None

    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    whole_caption = data.get("whole_caption", "")
    if not whole_caption:
        return None

    whole_image_path = os.path.join(sample_path, "whole_image.png")
    return (sample_name, whole_caption, whole_image_path)


def process_gpu_shard(gpu_id, sample_names, batch_size, skip_existing=False):
    """Process a shard of samples on a specific GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model on {device}...")
    model, processor = load_model(device)
    print(f"[GPU {gpu_id}] Checking {len(sample_names)} samples (skip_existing={skip_existing})...")

    pending = []
    for sn in tqdm(sample_names, desc=f"[GPU {gpu_id}] Scanning", leave=False):
        result = process_sample_check(sn, skip_existing=skip_existing)
        if result:
            pending.append(result)

    skipped = len(sample_names) - len(pending)
    print(f"[GPU {gpu_id}] {len(pending)} to process, {skipped} already done")

    processed = 0
    pbar = tqdm(total=len(pending), desc=f"[GPU {gpu_id}] Captioning")
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        names = [b[0] for b in batch]
        captions = [b[1] for b in batch]
        img_paths = [b[2] for b in batch]

        try:
            refined = refine_caption_batch(model, processor, captions, img_paths, device)
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Batch error at {names[0]}: {e}")
            refined = [None] * len(batch)

        for sn, ref_caption in zip(names, refined):
            if ref_caption is None:
                continue
            sample_path = os.path.join(ROOT_DIR, sn)
            metadata_old_path = os.path.join(sample_path, "metadata_old.json")
            metadata_path = os.path.join(sample_path, "metadata.json")

            with open(metadata_old_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["whole_caption"] = ref_caption
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            processed += 1

        pbar.update(len(batch))

    pbar.close()
    print(f"[GPU {gpu_id}] Done. Processed {processed} samples.")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start from this sample index (e.g. 100000 to skip first 100k)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='End at this sample index (exclusive). Default: all samples')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Override ROOT_DIR')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU (default: 8)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip already-processed samples (for resuming interrupted runs)')
    args = parser.parse_args()

    global ROOT_DIR
    if args.root_dir:
        ROOT_DIR = args.root_dir

    print(f"Scanning {ROOT_DIR} ...")
    all_entries = os.listdir(ROOT_DIR)
    print(f"  Found {len(all_entries)} entries, filtering sample_ directories...")
    all_samples = sorted([d for d in all_entries if d.startswith("sample_")])
    print(f"  {len(all_samples)} sample directories found")

    end_idx = args.end_index if args.end_index else len(all_samples)
    all_samples = all_samples[args.start_index:end_idx]

    num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()

    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"Model: {QWEN_MODEL_PATH}")
    print(f"Samples to process: {len(all_samples)} (index {args.start_index} to {end_idx})")
    print(f"GPUs: {num_gpus}, Batch size: {args.batch_size}, Skip existing: {args.skip_existing}")

    if num_gpus > 1:
        print("Pre-downloading model to cache (avoids race condition across workers)...")
        from huggingface_hub import snapshot_download
        snapshot_download(QWEN_MODEL_PATH)
        print("Model cached. Launching workers...")

    if num_gpus == 1:
        process_gpu_shard(0, all_samples, args.batch_size, args.skip_existing)
    else:
        shard_size = (len(all_samples) + num_gpus - 1) // num_gpus
        shards = [all_samples[i * shard_size:(i + 1) * shard_size] for i in range(num_gpus)]

        from torch.multiprocessing import spawn
        spawn(_spawn_worker, args=(shards, args.batch_size, args.skip_existing), nprocs=num_gpus, join=True)


def _spawn_worker(gpu_id, shards, batch_size, skip_existing):
    process_gpu_shard(gpu_id, shards[gpu_id], batch_size, skip_existing)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Done! Total time: {time.time() - start_time:.2f} seconds")
