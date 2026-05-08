import os, yaml, random
import torch
import numpy as np
from typing import Union
import pickle
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipeline
from models.multiLayer_adapter import MultiLayerAdapter

def save_checkpoint(transformer, multiLayer_adater, optimizer, optimizer_adapter, scheduler, scheduler_adapter, step, save_dir):
    import gc
    
    trans_dir = os.path.join(save_dir, "transformer")
    adapter_dir = os.path.join(save_dir, "adapter")
    os.makedirs(trans_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    # Get state dicts and IMMEDIATELY move to CPU to avoid GPU memory buildup
    flux_transformer_lora_state_dict = get_peft_model_state_dict(transformer)
    flux_transformer_lora_state_dict = {k: v.detach().cpu().to(torch.float32) for k, v in flux_transformer_lora_state_dict.items()}
    
    flux_adapter_lora_state_dict = get_peft_model_state_dict(multiLayer_adater)
    flux_adapter_lora_state_dict = {k: v.detach().cpu().to(torch.float32) for k, v in flux_adapter_lora_state_dict.items()}

    CustomFluxPipeline.save_lora_weights(
        os.path.join(trans_dir),
        flux_transformer_lora_state_dict,
        safe_serialization=True,
    )
    # Clear after saving
    del flux_transformer_lora_state_dict
    
    CustomFluxPipeline.save_lora_weights(
        os.path.join(adapter_dir),
        flux_adapter_lora_state_dict,
        safe_serialization=True,
    )
    # Clear after saving
    del flux_adapter_lora_state_dict

    torch.save({"layer_pe": transformer.layer_pe.detach().cpu().to(torch.float32)}, os.path.join(save_dir, "layer_pe.pth"))

    torch.save(optimizer.state_dict(), os.path.join(trans_dir, "optimizer.bin"))
    torch.save(optimizer_adapter.state_dict(), os.path.join(adapter_dir, "optimizer.bin"))

    torch.save(scheduler.state_dict(), os.path.join(trans_dir, "scheduler.bin"))
    torch.save(scheduler_adapter.state_dict(), os.path.join(adapter_dir, "scheduler.bin"))

    save_path = os.path.join(save_dir, f"random_states_0.pkl")
    state = {
        "step": step,
        "random_state": random.getstate(),
        "numpy_random_seed": np.random.get_state(),
        "torch_manual_seed": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()  # list of tensors

    with open(save_path, "wb") as f:
        pickle.dump(state, f)

    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[INFO] Saved RNG states + step {step} to {save_path}")
    

def load_checkpoint(transformer, multiLayer_adater, optimizer, optimizer_adapter, scheduler, scheduler_adapter, ckpt_dir, device="cuda"):
    trans_dir = os.path.join(ckpt_dir, "transformer")
    adapter_dir = os.path.join(ckpt_dir, "adapter")
    start_step = 0

    lora_path = os.path.join(trans_dir, "pytorch_lora_weights.safetensors")
    lora_path_adapter = os.path.join(adapter_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(lora_path):
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path)
        stripped = {k.replace("transformer.", "", 1) if k.startswith("transformer.") else k: v for k, v in lora_state_dict.items()}
        result = set_peft_model_state_dict(transformer, stripped)
        if result.unexpected_keys:
            print(f"[WARN] Transformer LoRA: {len(result.unexpected_keys)} unexpected keys")
        print(f"[INFO] Loaded Transformer LoRA weights ({len(stripped)} keys).")
    if os.path.exists(lora_path_adapter):
        lora_state_dict = CustomFluxPipeline.lora_state_dict(lora_path_adapter)
        stripped = {k.replace("transformer.", "", 1) if k.startswith("transformer.") else k: v for k, v in lora_state_dict.items()}
        result = set_peft_model_state_dict(multiLayer_adater, stripped)
        if result.unexpected_keys:
            print(f"[WARN] Adapter LoRA: {len(result.unexpected_keys)} unexpected keys")
        print(f"[INFO] Loaded Adapter LoRA weights ({len(stripped)} keys).")

    pe_path = os.path.join(ckpt_dir, "layer_pe.pth")
    if os.path.exists(pe_path):
        layer_pe = torch.load(pe_path)
        missing_keys, unexpected_keys = transformer.load_state_dict(layer_pe, strict=False)

    opt_path = os.path.join(trans_dir, "optimizer.bin")
    opt_path_adapter = os.path.join(adapter_dir, "optimizer.bin")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print("[INFO] Loaded optimizer state.")
    if os.path.exists(opt_path_adapter):
        optimizer_adapter.load_state_dict(torch.load(opt_path_adapter, map_location=device))
        print("[INFO] Loaded optimizer state.")

    sch_path = os.path.join(trans_dir, "scheduler.bin")
    sch_path_adapter = os.path.join(adapter_dir, "scheduler.bin")
    if os.path.exists(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        print("[INFO] Loaded scheduler state.")
    if os.path.exists(sch_path_adapter):
        scheduler_adapter.load_state_dict(torch.load(sch_path_adapter, map_location=device))
        print("[INFO] Loaded scheduler state.")

    rng_file = None
    for f in os.listdir(ckpt_dir):
        if f.startswith("random_states_") and f.endswith(".pkl"):
            rng_file = os.path.join(ckpt_dir, f)
            break

    if rng_file:
        with open(rng_file, "rb") as f:
            state = pickle.load(f)
        start_step = state.get("step", 0)

        if "random_state" in state:
            random.setstate(state["random_state"])
        if "numpy_random_seed" in state:
            np.random.set_state(state["numpy_random_seed"])
        if "torch_manual_seed" in state:
            torch.set_rng_state(state["torch_manual_seed"])
        if "torch_cuda_manual_seed" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_manual_seed"])

        print(f"[INFO] Resumed RNG states + step {start_step}")

    return start_step


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_input_box(layer_boxes, image_size=512):
    """
    Quantize layer boxes to 16-pixel grid for latent space alignment.
    
    Args:
        layer_boxes: List of boxes in xyxy format [x0, y0, x1, y1]
        image_size: Image size to clamp bounds (default 512)
    
    Returns:
        List of quantized boxes in xyxy format
    """
    list_layer_box = []
    for layer_box in layer_boxes:
        min_col, min_row = layer_box[0], layer_box[1]
        max_col, max_row = layer_box[2], layer_box[3]
        
        # Floor for min (start of box)
        quantized_min_row = (min_row // 16) * 16
        quantized_min_col = (min_col // 16) * 16
        
        # Ceiling for max (end of box) - use (val + 15) // 16 * 16 for proper ceiling
        quantized_max_row = ((max_row + 15) // 16) * 16
        quantized_max_col = ((max_col + 15) // 16) * 16
        
        # Clamp to image bounds
        quantized_min_row = max(0, quantized_min_row)
        quantized_min_col = max(0, quantized_min_col)
        quantized_max_row = min(image_size, quantized_max_row)
        quantized_max_col = min(image_size, quantized_max_col)
        
        # Ensure minimum box size of 16 pixels (1 latent token) in each dimension
        # This prevents zero-size boxes that cause reshape errors
        if quantized_max_col <= quantized_min_col:
            # Expand the box, preferring to expand max if there's room
            if quantized_min_col + 16 <= image_size:
                quantized_max_col = quantized_min_col + 16
            else:
                quantized_min_col = max(0, quantized_max_col - 16)
                quantized_max_col = quantized_min_col + 16
        
        if quantized_max_row <= quantized_min_row:
            # Expand the box, preferring to expand max if there's room
            if quantized_min_row + 16 <= image_size:
                quantized_max_row = quantized_min_row + 16
            else:
                quantized_min_row = max(0, quantized_max_row - 16)
                quantized_max_row = quantized_min_row + 16

        list_layer_box.append((quantized_min_col, quantized_min_row, quantized_max_col, quantized_max_row))
    return list_layer_box


def set_lora_into_transformer(
    model: Union[CustomFluxTransformer2DModel, MultiLayerAdapter],
    lora_rank: int,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.1,
):

    target_modules = [
        "to_k", "to_q", "to_v",
        "to_out.0",
        "add_k_proj", "add_q_proj", "add_v_proj",
        "to_add_out",
    ] + [f"single_transformer_blocks.{i}.proj_out" for i in range(model.config.num_single_layers)] + [f"transformer_blocks.{i}.proj_out" for i in range(model.config.num_layers)]

    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    model.add_adapter(transformer_lora_config)
    return model


def build_layer_mask(n_layers, H_lat, W_lat, list_layer_box):
    mask = torch.zeros((n_layers, 1, H_lat, W_lat), dtype=torch.float32)
    for i, box in enumerate(list_layer_box):
        if box is None: 
            continue
        x1, y1, x2, y2 = box
        x1_t, y1_t, x2_t, y2_t = x1 // 8, y1 // 8, x2 // 8, y2 // 8
        x1_t, y1_t = max(0, x1_t), max(0, y1_t)
        x2_t, y2_t = min(W_lat, x2_t), min(H_lat, y2_t)
        if x2_t > x1_t and y2_t > y1_t:
            mask[i, :, y1_t:y2_t, x1_t:x2_t] = 1.0
    return mask


def encode_target_latents(pipeline, pixel_bchw, n_layers, list_layer_box):
    device = pixel_bchw.device
    dtype  = pixel_bchw.dtype

    vae = pipeline.vae.eval()
    bs, n_layers_in, C, H, W = pixel_bchw.shape
    assert n_layers_in == n_layers, f"The number of input layers {n_layers_in} does not match the specified number of layers {n_layers}"

    with torch.no_grad():
        dummy_lat = vae.encode(pixel_bchw[:,0]).latent_dist.sample()
    _, C_lat, H_lat, W_lat = dummy_lat.shape

    x0 = torch.zeros((bs, n_layers, C_lat, H_lat, W_lat), device=device, dtype=dtype)

    with torch.no_grad():
        for i in range(n_layers):
            pixel_i = pixel_bchw[:, i]
            lat = vae.encode(pixel_i).latent_dist.sample()  # [1,C_lat,H_lat,W_lat]
            lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
            x0[:, i] = lat

    latent_ids = pipeline._prepare_latent_image_ids(H_lat, W_lat, list_layer_box, device, dtype)

    return x0, latent_ids


def get_timesteps(pipeline, image_seq_len, num_inference_steps, device):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.base_image_seq_len,
        pipeline.scheduler.config.max_image_seq_len,
        pipeline.scheduler.config.base_shift,
        pipeline.scheduler.config.max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler=pipeline.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )

    return timesteps


# ============================================================================
# Box utilities for Prism blended dataset
# ============================================================================

def scale_box_xyxy(box, source_size: int, target_size: int):
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


def quantize_box_16(box, target_size: int):
    """
    Quantize box to 16-pixel grid for latent space alignment.
    Box is in xyxy format.
    """
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


def get_prism_layer_boxes_xyxy(layers, source_size: int, target_size: int):
    """
    Extract and scale layer boxes from prism blended metadata.
    
    Note: Our blended dataset uses xyxy format [x0, y0, x1, y1].
    
    Args:
        layers: List of layer metadata dicts with 'box' field (xyxy format)
        source_size: Size the data was generated at (e.g., 512)
        target_size: Size to run inference at (e.g., 512)
    
    Returns:
        List of quantized boxes in xyxy format
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


def xywh_to_xyxy(box):
    """Convert (x, y, w, h) to (x0, y0, x1, y1)."""
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box):
    """Convert (x0, y0, x1, y1) to (x, y, w, h)."""
    x0, y0, x1, y1 = box
    return (x0, y0, x1 - x0, y1 - y0)