import os
import sys
import logging

# Add project root to Python path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress CLIP tokenizer truncation warnings (CLIP truncates internally, T5 handles full caption)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict

from models.multiLayer_adapter import MultiLayerAdapter
from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipeline, CustomFluxPipelineCfgLayer
from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE
from tools.tools import load_config, seed_everything


def load_prism_metadata(jsonl_path: str):
    """Load prism blend metadata from JSONL file."""
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def scale_box_xyxy(box, source_size: int, target_size: int) -> tuple:
    """
    Scale a box from source_size to target_size.
    Box is already in xyxy format: [x0, y0, x1, y1].
    
    Args:
        box: [x0, y0, x1, y1] in source_size coordinates
        source_size: Original data size (e.g., 512)
        target_size: Target inference size (e.g., 512)
    
    Returns:
        (x0, y0, x1, y1) in target_size coordinates
    """
    scale = target_size / source_size
    x0, y0, x1, y1 = box
    
    x0_s = int(x0 * scale)
    y0_s = int(y0 * scale)
    x1_s = int(x1 * scale)
    y1_s = int(y1 * scale)
    
    # Clamp to valid range
    x0_s = max(0, x0_s)
    y0_s = max(0, y0_s)
    x1_s = min(target_size, x1_s)
    y1_s = min(target_size, y1_s)
    
    return (x0_s, y0_s, x1_s, y1_s)


def quantize_box_16(box: tuple, target_size: int) -> tuple:
    x0, y0, x1, y1 = box
    
    # Quantize to 16-pixel grid
    x0_q = (x0 // 16) * 16
    y0_q = (y0 // 16) * 16
    x1_q = ((x1 + 15) // 16) * 16
    y1_q = ((y1 + 15) // 16) * 16
    
    # Clamp to image bounds
    x0_q = max(0, x0_q)
    y0_q = max(0, y0_q)
    x1_q = min(target_size, x1_q)
    y1_q = min(target_size, y1_q)
    
    return (x0_q, y0_q, x1_q, y1_q)


def get_layer_boxes(layers: list, source_size: int, target_size: int) -> list:
    """
    Extract and scale layer boxes from metadata.
    Boxes in metadata are in xyxy format: [x0, y0, x1, y1].
    Returns list of quantized boxes in xyxy format.
    """
    boxes = []
    
    for layer in layers:
        box = layer.get('box', [0, 0, source_size, source_size])
        
        # Scale from source to target size (box is already xyxy)
        scaled_box = scale_box_xyxy(box, source_size, target_size)
        
        # Quantize to 16-pixel grid
        quantized_box = quantize_box_16(scaled_box, target_size)
        
        boxes.append(quantized_box)
    
    return boxes


def initialize_pipeline(config):
    """Initialize the SynLayers pipeline with all components."""
    
    print("[INFO] Loading Transparent VAE...", flush=True)
    vae_args = argparse.Namespace(
        max_layers=config.get('max_layers', 48),
        decoder_arch=config.get('decoder_arch', 'vit'),
        pos_embedding=config.get('pos_embedding', 'rope'),
        layer_embedding=config.get('layer_embedding', 'rope'),
        single_layer_decoder=config.get('single_layer_decoder', None)
    )
    transp_vae = CustomVAE(vae_args)
    transp_vae_path = config.get('transp_vae_path')
    transp_vae_weights = torch.load(transp_vae_path, map_location=torch.device("cuda"))
    missing_keys, unexpected_keys = transp_vae.load_state_dict(transp_vae_weights['model'], strict=False)
    if missing_keys:
        print(f"ViT Encoder Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"ViT Encoder Unexpected keys: {unexpected_keys}")
    transp_vae.eval()
    transp_vae = transp_vae.to(torch.device("cuda"))
    print("[INFO] Transparent VAE loaded.", flush=True)

    print("[INFO] Loading pretrained Transformer model...", flush=True)
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.get('transformer_varient', config['pretrained_model_name_or_path']),
        subfolder="" if 'transformer_varient' in config else "transformer",
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    )

    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = config['max_layer_num']
    mmdit_config = FrozenDict(mmdit_config)

    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)

    # Load LoRA weights
    if 'pretrained_lora_dir' in config:
        print("[INFO] Loading pretrained LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['pretrained_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()

    if 'artplus_lora_dir' in config:
        print("[INFO] Loading artplus LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['artplus_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()

    # Load layer_pe weights
    layer_pe_path = os.path.join(config['layer_ckpt'], "layer_pe.pth")
    if os.path.exists(layer_pe_path):
        print(f"[INFO] Loading layer_pe from {layer_pe_path}...", flush=True)
        layer_pe = torch.load(layer_pe_path)
        transformer.load_state_dict(layer_pe, strict=False)

    # Load MultiLayer-Adapter
    print("[INFO] Loading MultiLayer-Adapter...", flush=True)
    multiLayer_adapter = MultiLayerAdapter.from_pretrained(config['pretrained_adapter_path']).to(torch.bfloat16).to(torch.device("cuda"))
    
    if 'adapter_lora_dir' in config:
        print("[INFO] Loading adapter LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['adapter_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, multiLayer_adapter)
        multiLayer_adapter.fuse_lora(safe_fusing=True)
        multiLayer_adapter.unload_lora()
    
    multiLayer_adapter.set_layerPE(transformer.layer_pe, transformer.max_layer_num)

    # Initialize pipeline
    pipeline = CustomFluxPipelineCfgLayer.from_pretrained(
        config['pretrained_model_name_or_path'],
        transformer=transformer,
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    ).to(torch.device("cuda"))
    pipeline.set_multiLayerAdapter(multiLayer_adapter)

    # Load trained LoRA
    if 'lora_ckpt' in config:
        print(f"[INFO] Loading trained LoRA from {config['lora_ckpt']}...", flush=True)
        pipeline.load_lora_weights(config['lora_ckpt'], adapter_name="layer")

    return pipeline, transp_vae


def load_adapter_image(sample_metadata: dict, target_size: int, data_dir: str = None) -> Image.Image:
    """
    Load and resize the whole_image as adapter input.
    
    New dataset structure uses:
    - sample_dir/whole_image.png (the composite image)
    """
    img = None
    
    # Try sample_dir + whole_image.png (new dataset format)
    sample_dir = sample_metadata.get('sample_dir', '')
    if sample_dir:
        # sample_dir might be relative (e.g., "sample_000000") or absolute
        if data_dir and not os.path.isabs(sample_dir):
            full_sample_dir = os.path.join(data_dir, sample_dir)
        else:
            full_sample_dir = sample_dir
        
        whole_img_path = os.path.join(full_sample_dir, 'whole_image.png')
        if os.path.exists(whole_img_path):
            img = Image.open(whole_img_path).convert('RGB')
    
    # Fallback to blend_path if exists (old format compatibility)
    if img is None:
        blend_path = sample_metadata.get('blend_path', '')
        if blend_path and os.path.exists(blend_path):
            img = Image.open(blend_path).convert('RGB')
    
    # Final fallback: create gray image
    if img is None:
        img = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    
    # Resize to target size
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    
    return img


@torch.no_grad()
def inference_prism(config):
    """Main inference function for prism blended dataset."""
    
    if config.get('seed') is not None:
        seed_everything(config['seed'])
    
    # Get sizes
    source_size = config.get('source_size', 1024)  # Data generated at this size
    target_size = config.get('target_size', 512)   # Inference at this size
    
    print(f"[INFO] Source size: {source_size}, Target size: {target_size}", flush=True)
    
    # Create output directory
    save_dir = config.get('save_dir', './prism_inference_output')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "merged"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "merged_rgba"), exist_ok=True)
    
    # Load pipeline
    pipeline, transp_vae = initialize_pipeline(config)
    
    # Load test data
    test_jsonl = config.get('test_jsonl', '')
    if not test_jsonl or not os.path.exists(test_jsonl):
        raise ValueError(f"Test JSONL not found: {test_jsonl}")
    
    all_samples = load_prism_metadata(test_jsonl)
    total_available = len(all_samples)

    # Determine sample range (1-based: sample 1 = first sample in jsonl)
    start_idx = config.get('start_idx', 1)
    end_idx = config.get('end_idx', total_available)
    max_samples = config.get('max_samples', None)

    if max_samples and not config.get('end_idx'):
        end_idx = min(start_idx + max_samples - 1, total_available)

    start_idx = max(1, min(start_idx, total_available))
    end_idx = max(start_idx, min(end_idx, total_available))

    samples = all_samples[start_idx - 1 : end_idx]

    print(f"[INFO] Total samples in dataset: {total_available}", flush=True)
    print(f"[INFO] Processing samples {start_idx} to {end_idx} ({len(samples)} samples)", flush=True)
    
    generator = torch.Generator(device=torch.device("cuda")).manual_seed(config.get('seed', 42))
    
    for local_idx, sample in enumerate(samples):
        idx = start_idx - 1 + local_idx
        sample_name = sample.get('sample_dir', f'sample_{idx:06d}')
        print(f"Processing [{local_idx+1}/{len(samples)}] idx={idx} ({sample_name})...", flush=True)
        
        # Get layers and scale boxes
        layers = sample.get('layers', [])
        layer_boxes = get_layer_boxes(layers, source_size, target_size)
        
        # Add whole image box and background box at the beginning
        whole_box = (0, 0, target_size, target_size)
        bg_box = (0, 0, target_size, target_size)
        
        # Full layer structure: [whole_image, background, layer_0, layer_1, layer_2, ...]
        # layers list contains layer_00, layer_01, layer_02, ... (all foreground layers)
        # We prepend whole_image and base_image boxes
        all_boxes = [whole_box, bg_box] + layer_boxes
        
        print(f"  Size: {target_size}x{target_size}, Layers: {len(all_boxes)}", flush=True)
        
        # Load adapter image (whole_image.png from sample directory)
        data_dir = config.get('data_dir', '')
        adapter_img = load_adapter_image(sample, target_size, data_dir=data_dir)
        
        # Get caption (CLIP auto-truncates to 77 tokens, T5 handles full caption)
        caption = sample.get('whole_caption', '')
        
        # Run pipeline
        try:
            x_hat, image, _ = pipeline(
                prompt=caption,
                adapter_image=adapter_img,
                adapter_conditioning_scale=config.get('adapter_scale', 0.9),
                validation_box=all_boxes,
                generator=generator,
                height=target_size,
                width=target_size,
                guidance_scale=config.get('cfg', 4.0),
                num_layers=len(all_boxes),
                sdxl_vae=transp_vae,
            )
        except Exception as e:
            print(f"  Error during inference: {e}", flush=True)
            continue
        
        # Process output
        x_hat = (x_hat + 1) / 2  # [-1,1] -> [0,1]
        x_hat = x_hat.squeeze(0).permute(1, 0, 2, 3).to(torch.float32)
        
        # Create case directory
        case_dir = os.path.join(save_dir, sample_name)
        os.makedirs(case_dir, exist_ok=True)
        
        # Save whole image RGBA (x_hat[0])
        whole_image_layer = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(whole_image_layer, "RGBA").save(os.path.join(case_dir, "whole_image_rgba.png"))
        
        # Save background RGBA (x_hat[1])
        background_layer = (x_hat[1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(background_layer, "RGBA").save(os.path.join(case_dir, "background_rgba.png"))
        
        # Save original input
        adapter_img.save(os.path.join(case_dir, "origin.png"))
        
        # Save individual layers (x_hat[2:])
        merged_image = image[1]  # Background as base
        
        for layer_idx in range(2, x_hat.shape[0]):
            rgba_layer = (x_hat[layer_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgba_image = Image.fromarray(rgba_layer, "RGBA")
            rgba_image.save(os.path.join(case_dir, f"layer_{layer_idx - 2}_rgba.png"))
            
            # Composite onto merged image
            merged_image = Image.alpha_composite(merged_image.convert('RGBA'), rgba_image)
        
        # Save merged results
        merged_image.convert('RGB').save(os.path.join(save_dir, "merged", f"{sample_name}.png"))
        merged_image.convert('RGB').save(os.path.join(case_dir, "merged.png"))
        merged_image.save(os.path.join(save_dir, "merged_rgba", f"{sample_name}.png"))
        
        # Save metadata for this case
        case_meta = {
            'sample_idx': idx,
            'sample_name': sample_name,
            'target_size': target_size,
            'source_size': source_size,
            'num_layers': len(all_boxes),
            'boxes': all_boxes,
            'caption': caption,
        }
        with open(os.path.join(case_dir, 'inference_meta.json'), 'w') as f:
            json.dump(case_meta, f, indent=2)
        
        # Clean up GPU memory periodically
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"[INFO] Inference complete. Results saved to {save_dir}", flush=True)
    
    # Cleanup
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="1-based start index. E.g. 571 means start from the 571st sample.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="1-based end index (inclusive). E.g. 1000 means stop at the 1000th sample.")
    args = parser.parse_args()
    
    config = load_config(args.config_path)
    if args.start_idx is not None:
        config['start_idx'] = args.start_idx
    if args.end_idx is not None:
        config['end_idx'] = args.end_idx
    inference_prism(config)

