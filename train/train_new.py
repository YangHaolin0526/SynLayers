import os, random
import sys
import argparse
import logging

# Add project root to Python path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress CLIP tokenizer truncation warnings (CLIP truncates internally, T5 handles full caption)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
from prodigyopt import Prodigy
from diffusers import FluxTransformer2DModel
from diffusers.configuration_utils import FrozenDict

from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipelineCfgLayer, CustomFluxPipeline
from models.multiLayer_adapter import MultiLayerAdapter
from tools.tools import save_checkpoint, load_checkpoint, load_config, seed_everything, get_input_box, set_lora_into_transformer, build_layer_mask, encode_target_latents, get_timesteps
from tools.dataset import PrismBlendDataset, prism_collate_fn


def train(config_path, multi_gpu=False):
    config = load_config(config_path)
    seed_everything(config.get("seed", 1234))

    # ── Device setup ──────────────────────────────────────────────────
    if multi_gpu:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = (local_rank == 0)
        if is_main:
            print(f"[INFO] Multi-GPU DDP mode: {world_size} GPUs", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        is_main = True
        print(f"[INFO] Single GPU mode: device={device}", flush=True)

    # ── Load Transformer ──────────────────────────────────────────────
    if is_main: print("[INFO] Loading pretrained Transformer...", flush=True)

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
    missing_keys, unexpected_keys = transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    if is_main:
        if missing_keys: print(f"[WARN] Missing keys: {missing_keys}")
        if unexpected_keys: print(f"[WARN] Unexpected keys: {unexpected_keys}")

    del transformer_orig
    torch.cuda.empty_cache()

    if 'pretrained_lora_dir' in config:
        if is_main: print("[INFO] Loading LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['pretrained_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()

    if 'artplus_lora_dir' in config:
        if is_main: print("[INFO] Loading artplus LoRA weights...", flush=True)
        lora_state_dict = CustomFluxPipeline.lora_state_dict(config['artplus_lora_dir'])
        CustomFluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora()

    # ── Load Adapter ──────────────────────────────────────────────────
    if is_main: print("[INFO] Loading MultiLayer-Adapter weights...", flush=True)
    multiLayer_adapter = MultiLayerAdapter.from_pretrained(config['pretrained_adapter_path']).to(torch.bfloat16).to(device)
    multiLayer_adapter.set_layerPE(transformer.layer_pe, transformer.max_layer_num)

    # ── Load Pipeline ─────────────────────────────────────────────────
    pipeline = CustomFluxPipelineCfgLayer.from_pretrained(
        config['pretrained_model_name_or_path'],
        transformer=transformer,
        revision=config.get('revision', None),
        variant=config.get('variant', None),
        torch_dtype=torch.bfloat16,
        cache_dir=config.get('cache_dir', None),
    ).to(device)
    pipeline.set_multiLayerAdapter(multiLayer_adapter)
    pipeline.transformer.gradient_checkpointing = True
    pipeline.multiLayerAdapter.gradient_checkpointing = True

    if is_main: print("[INFO] Pipeline loaded successfully.", flush=True)

    # ── LoRA injection ────────────────────────────────────────────────
    lora_rank = int(config.get("lora_rank", 16))
    lora_alpha = float(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.0))
    set_lora_into_transformer(pipeline.transformer, lora_rank, lora_alpha, lora_dropout)
    set_lora_into_transformer(pipeline.multiLayerAdapter, lora_rank, lora_alpha, lora_dropout)
    pipeline.transformer.requires_grad_(False)
    pipeline.multiLayerAdapter.requires_grad_(False)
    pipeline.transformer.train()
    pipeline.multiLayerAdapter.train()
    for n, param in pipeline.transformer.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for n, param in pipeline.multiLayerAdapter.named_parameters():
        if 'lora' in n or 'layer_pe' in n:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
    n_trainable_adapter = sum(p.numel() for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad)
    if is_main:
        print(f"[INFO] LoRA injected. Transformer Trainable params: {n_trainable/1e6:.2f}M; MultiLayer-Adapter Trainable params: {n_trainable_adapter/1e6:.2f}M", flush=True)

    # ── Multi-GPU setup ─────────────────────────────────────────────
    # NOTE: DDP wrapping conflicts with gradient_checkpointing (use_reentrant=False)
    # in this PyTorch version. Instead, we skip DDP and manually allreduce gradients
    # after backward. This is fully correct and avoids all hook conflicts.
    transformer_ddp = pipeline.transformer
    adapter_ddp = pipeline.multiLayerAdapter
    if multi_gpu and is_main:
        print("[INFO] Multi-GPU: manual gradient sync (no DDP wrappers).", flush=True)

    # ── Optimizers (follows original train.py exactly) ────────────────
    if is_main: print("[INFO] Using Prodigy optimizer.", flush=True)
    params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    params_adapter = [p for p in pipeline.multiLayerAdapter.parameters() if p.requires_grad]
    optimizer = Prodigy(
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    optimizer_adapter = Prodigy(
        params_adapter,
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.001,
        decouple=True,
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    scheduler_adapter = LambdaLR(optimizer_adapter, lr_lambda=lambda step: 1.0)

    # ── Dataset & DataLoader ──────────────────────────────────────────
    dataset = PrismBlendDataset(
        data_dir=config['data_dir'],
        jsonl_path=config.get('train_jsonl'),
        target_size=config.get('target_size', 512),
        split="train",
        max_layer_num=config.get('max_layer_num', None)
    )
    if multi_gpu:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=prism_collate_fn)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=prism_collate_fn)
    if is_main: print(f"[INFO] Dataset: {len(dataset)} samples", flush=True)

    # ── Training setup ────────────────────────────────────────────────
    max_steps = int(config.get("max_steps", 1000))
    log_every = int(config.get("log_every", 50))
    save_every = int(config.get("save_every", 500))
    accum_steps = int(config.get("accum_steps", 1))
    out_dir = config.get("output_dir", "./prism_train_out")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    tb_writer = SummaryWriter(out_dir) if is_main else None

    num_inference_steps = config.get("num_inference_steps", 28)

    start_step = 0
    if "resume_from" in config and config["resume_from"] is not None:
        ckpt_dir = config["resume_from"]
        start_step = load_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, ckpt_dir, device)

    pbar = tqdm(total=max_steps, desc="train", initial=start_step) if is_main else None
    step = start_step
    epoch = 0

    # ── Training loop (matches original train.py structure) ───────────
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            if step >= max_steps: break

            pixel_RGB = batch["pixel_RGB"].to(device=device, dtype=torch.bfloat16)
            pixel_RGB = pipeline.image_processor.preprocess(pixel_RGB[0])
            H = int(batch["height"][0])
            W = int(batch["width"][0])
            adapter_img = batch["whole_img"][0]
            caption = batch["caption"][0]
            layer_boxes = get_input_box(batch["layout"][0])

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=caption,
                    prompt_2=None,
                    num_images_per_prompt=1,
                    max_sequence_length=int(config.get("max_sequence_length", 512)),
                )

                prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)
                text_ids = text_ids.to(device=device, dtype=torch.bfloat16)

                adapter_image, _, _ = pipeline.prepare_image(
                    image=adapter_img,
                    width=W,
                    height=H,
                    batch_size=1,
                    num_images_per_prompt=1,
                    device=device,
                    dtype=pipeline.transformer.dtype,
                )

            x0, latent_image_ids = encode_target_latents(pipeline, pixel_RGB.unsqueeze(0), n_layers=len(layer_boxes), list_layer_box=layer_boxes)
            _, L, C_lat, H_lat, W_lat = x0.shape

            x1 = torch.randn_like(x0)
            image_seq_len = latent_image_ids.shape[0]
            timesteps = get_timesteps(pipeline, image_seq_len=image_seq_len, num_inference_steps=num_inference_steps, device=device)
            t = timesteps[random.randint(0, len(timesteps)-1)].to(device=device, dtype=torch.float32)
            t = t.expand(x0.shape[0]).to(x0.dtype)
            t_b = t.view(1, 1, 1, 1, 1).to(x0.dtype)
            t_b = t_b / 1000.0
            xt = (1.0 - t_b) * x0 + t_b * x1
            v_star = x1 - x0

            mask = build_layer_mask(L, H_lat, W_lat, layer_boxes).to(device=device, dtype=x0.dtype)
            mask = mask.unsqueeze(0)

            # classifier-free guidance
            guidance_scale = config.get('cfg', 4.0)
            if pipeline.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(x0.shape[0])
            else:
                guidance = None

            pipeline.transformer.train()
            pipeline.multiLayerAdapter.train()

            (
                adapter_block_samples,
                adapter_single_block_samples,
            ) = adapter_ddp(
                hidden_states=xt,
                list_layer_box=layer_boxes,
                adapter_cond=adapter_image,
                conditioning_scale=config.get("adapter_scale", 1.0),
                timestep=t / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )

            v_pred = transformer_ddp(
                hidden_states=xt,
                adapter_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_block_samples
                ],
                adapter_single_block_samples=[
                    sample.to(dtype=pipeline.transformer.dtype)
                    for sample in adapter_single_block_samples
                ] if adapter_single_block_samples is not None else adapter_single_block_samples,
                list_layer_box=layer_boxes,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=t / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            # MSE (masked)
            mse = (v_pred - v_star) ** 2
            mse = mse.mean(dim=2, keepdim=True)
            loss = (mse * mask).sum() / (mask.sum() + 1e-8)

            loss = loss / accum_steps
            loss.float().backward()

            if (step + 1) % accum_steps == 0:
                # Sync gradients across GPUs only at the accumulation boundary,
                # so each GPU accumulates locally for accum_steps before averaging.
                if multi_gpu:
                    for p in params + params_adapter:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_(params, max_norm=float(config.get("grad_clip", 1.0)))
                optimizer.step()
                optimizer_adapter.step()
                scheduler.step()
                scheduler_adapter.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_adapter.zero_grad(set_to_none=True)

                if is_main:
                    tb_writer.add_scalar("loss", loss.item(), step)

            step += 1
            if is_main and step % log_every == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
                pbar.update(log_every)

            if (step % save_every == 0 or step == max_steps) and is_main:
                step_dir = os.path.join(out_dir, f"step_{step}")
                os.makedirs(step_dir, exist_ok=True)
                save_checkpoint(pipeline.transformer, pipeline.multiLayerAdapter, optimizer, optimizer_adapter, scheduler, scheduler_adapter, step, step_dir)
                print(f"[INFO] Checkpoint saved to {step_dir}", flush=True)

        epoch += 1

    if is_main:
        pbar.close()
        print("[DONE] Training finished.")

    if multi_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--multi_gpu", "--model_parallel", action="store_true",
                        help="Use multi-GPU DDP training")
    args = parser.parse_args()
    train(args.config_path, multi_gpu=args.multi_gpu)
