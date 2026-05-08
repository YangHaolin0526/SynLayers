import os
import re
import json
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.inception import inception_v3
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def rgba_to_rgb_masked(rgba_img: np.ndarray) -> np.ndarray:
    rgba = rgba_img.astype(np.float32)
    if rgba.max() <= 1.0:
        rgba = rgba * 255.0

    rgb = rgba[..., :3]
    alpha = rgba[..., 3:] / 255.0
    background = np.ones_like(rgb) * 128.0
    out = rgb * alpha + background * (1 - alpha)
    return out.astype(rgba_img.dtype)


def extract_mask(img):
    if img.shape[-1] == 4:
        mask = img[..., 3] / 255.0
    else:
        mask = np.ones(img.shape[:2])
    return mask


def compute_iou(pred_mask, gt_mask, thresh=0.0):
    pred_bin = pred_mask > thresh
    gt_bin = gt_mask > thresh
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(intersection / (union + 1e-8))


def compute_mask_metrics(pred_mask, gt_mask, thresh=0.0):
    pred_bin = (pred_mask > thresh).astype(np.uint8).flatten()
    gt_bin = (gt_mask > thresh).astype(np.uint8).flatten()
    prec = precision_score(gt_bin, pred_bin, zero_division=0)
    rec = recall_score(gt_bin, pred_bin, zero_division=0)
    f1 = f1_score(gt_bin, pred_bin, zero_division=0)
    iou = compute_iou(pred_mask, gt_mask, thresh)
    return iou, prec, rec, f1


def load_gt_sample(gt_sample_dir: str):
    meta_path = os.path.join(gt_sample_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(gt_sample_dir, 'metadata_old.json')
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    composite = Image.open(os.path.join(gt_sample_dir, 'whole_image.png')).convert("RGBA")
    background = Image.open(os.path.join(gt_sample_dir, 'base_image.png')).convert("RGBA")

    layers_rgba = []
    layer_boxes = []
    for layer_info in meta.get('layers', []):
        img_path = os.path.join(gt_sample_dir, layer_info['image_path'])
        if os.path.exists(img_path):
            layers_rgba.append(Image.open(img_path).convert("RGBA"))
            layer_boxes.append(layer_info['box'])

    return composite, background, layers_rgba, layer_boxes


def load_pred_sample(pred_sample_dir: str, gt_boxes_fallback=None):
    merged_path = os.path.join(pred_sample_dir, 'merged.png')
    if not os.path.exists(merged_path):
        return None

    composite = Image.open(merged_path).convert("RGBA")

    bg_path = os.path.join(pred_sample_dir, 'background_rgba.png')
    if not os.path.exists(bg_path):
        return None
    background = Image.open(bg_path).convert("RGBA")

    layer_pngs = []
    for f_name in sorted(os.listdir(pred_sample_dir)):
        m = re.match(r'layer_(\d+)_rgba\.png', f_name)
        if m:
            layer_pngs.append((int(m.group(1)), f_name))
    layer_pngs.sort(key=lambda x: x[0])

    layers_rgba = []
    for _, fname in layer_pngs:
        layers_rgba.append(Image.open(os.path.join(pred_sample_dir, fname)).convert("RGBA"))

    meta_path = os.path.join(pred_sample_dir, 'inference_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f_meta:
            meta = json.load(f_meta)
        all_boxes = meta.get('boxes', [])
        layer_boxes = all_boxes[2:] if len(all_boxes) > 2 else []
    elif gt_boxes_fallback is not None:
        layer_boxes = gt_boxes_fallback
    else:
        layer_boxes = []

    return composite, background, layers_rgba, layer_boxes


def process_case(args):
    sample_name, gt_dir, pred_dir = args
    alpha_thresh = 0.1

    gt_sample_dir = os.path.join(gt_dir, sample_name)
    pred_sample_dir = os.path.join(pred_dir, sample_name)

    if not os.path.isdir(gt_sample_dir):
        return None

    gt_data = load_gt_sample(gt_sample_dir)
    if gt_data is None:
        return None

    gt_boxes_for_fallback = gt_data[3]
    pred_data = load_pred_sample(pred_sample_dir, gt_boxes_fallback=gt_boxes_for_fallback)
    if pred_data is None:
        return None

    gt_comp_pil, gt_bg_pil, gt_layers_pil, gt_boxes = gt_data
    pred_comp_pil, pred_bg_pil, pred_layers_pil, pred_boxes = pred_data

    gt_comp_rgb = rgba_to_rgb_masked(np.array(gt_comp_pil))
    pred_comp_rgb = rgba_to_rgb_masked(np.array(pred_comp_pil))

    case_comp_psnr = psnr(gt_comp_rgb, pred_comp_rgb, data_range=255)
    case_comp_ssim = ssim(gt_comp_rgb, pred_comp_rgb, channel_axis=2, data_range=255)

    case_layer_psnr, case_layer_ssim = [], []
    case_mask_metrics = []
    case_gt_imgs_fid, case_pred_imgs_fid = [], []

    n_layers = min(len(gt_layers_pil), len(pred_layers_pil))
    boxes_to_use = pred_boxes[:n_layers] if pred_boxes else gt_boxes[:n_layers]

    for i in range(n_layers):
        gt_layer_pil, pred_layer_pil = gt_layers_pil[i], pred_layers_pil[i]

        if i < len(boxes_to_use):
            box = boxes_to_use[i]
            gt_crop_pil = gt_layer_pil.crop(box)
            pred_crop_pil = pred_layer_pil.crop(box)
        else:
            gt_crop_pil, pred_crop_pil = gt_layer_pil, pred_layer_pil

        gt_crop_np, pred_crop_np = np.array(gt_crop_pil), np.array(pred_crop_pil)
        if gt_crop_np.size == 0 or pred_crop_np.size == 0:
            continue

        if gt_crop_np.shape != pred_crop_np.shape:
            h = min(gt_crop_np.shape[0], pred_crop_np.shape[0])
            w = min(gt_crop_np.shape[1], pred_crop_np.shape[1])
            gt_crop_np, pred_crop_np = gt_crop_np[:h, :w], pred_crop_np[:h, :w]

        gt_rgb, pred_rgb = rgba_to_rgb_masked(gt_crop_np), rgba_to_rgb_masked(pred_crop_np)

        case_layer_psnr.append(psnr(gt_rgb, pred_rgb, data_range=255))
        case_layer_ssim.append(ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255))

        gt_mask, pred_mask = extract_mask(gt_crop_np), extract_mask(pred_crop_np)
        case_mask_metrics.append(compute_mask_metrics(pred_mask, gt_mask, thresh=alpha_thresh))

        case_gt_imgs_fid.append(gt_rgb)
        case_pred_imgs_fid.append(pred_rgb)

    return (
        case_layer_psnr,
        case_layer_ssim,
        case_mask_metrics,
        case_comp_psnr,
        case_comp_ssim,
        case_gt_imgs_fid,
        case_pred_imgs_fid,
        gt_comp_rgb,
        pred_comp_rgb,
    )


class ImageListDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.img_list[idx]))


@torch.no_grad()
def get_inception_features(img_list, model, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = ImageListDataset(img_list, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    feats = []
    for batch in tqdm(dataloader, desc="  Calculating Inception Features", leave=False):
        batch = batch.to(device)
        feat = model(batch)
        feats.append(feat.cpu().numpy())

    return np.concatenate(feats, axis=0)


def calculate_fid(feats1, feats2):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def evaluate(gt_dir, pred_dir, output_dir, num_processes=64, fid_batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_samples = sorted([
        d for d in os.listdir(pred_dir)
        if d.startswith('sample_') and os.path.isdir(os.path.join(pred_dir, d))
    ])
    logger.info(f"Found {len(pred_samples)} samples. Starting parallel processing with {num_processes} workers...")

    all_layer_psnr, all_layer_ssim = [], []
    all_mask_metrics = []
    all_comp_psnr, all_comp_ssim = [], []
    gt_layers_for_fid, pred_layers_for_fid = [], []
    gt_comp_for_fid, pred_comp_for_fid = [], []

    tasks = [(s, gt_dir, pred_dir) for s in pred_samples]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks), desc="Processing samples"))

    for result in results:
        if result is None:
            continue
        (
            case_layer_psnr, case_layer_ssim, case_mask_metrics, case_comp_psnr,
            case_comp_ssim, case_gt_imgs_fid, case_pred_imgs_fid, gt_comp_rgb, pred_comp_rgb
        ) = result

        all_layer_psnr.extend(case_layer_psnr)
        all_layer_ssim.extend(case_layer_ssim)
        all_mask_metrics.extend(case_mask_metrics)
        all_comp_psnr.append(case_comp_psnr)
        all_comp_ssim.append(case_comp_ssim)
        gt_layers_for_fid.extend(case_gt_imgs_fid)
        pred_layers_for_fid.extend(case_pred_imgs_fid)
        gt_comp_for_fid.append(gt_comp_rgb)
        pred_comp_for_fid.append(pred_comp_rgb)

    logger.info("Loading Inception v3 (legacy eval.py style)...")
    inception = inception_v3(pretrained=True, transform_input=False).to(device).eval()

    fid_layers = calculate_fid(
        get_inception_features(gt_layers_for_fid, inception, device, fid_batch_size),
        get_inception_features(pred_layers_for_fid, inception, device, fid_batch_size),
    ) if len(gt_layers_for_fid) > 1 else float('nan')

    fid_comp = calculate_fid(
        get_inception_features(gt_comp_for_fid, inception, device, fid_batch_size),
        get_inception_features(pred_comp_for_fid, inception, device, fid_batch_size),
    ) if len(gt_comp_for_fid) > 1 else float('nan')

    mask_arr = np.array(all_mask_metrics) if all_mask_metrics else np.zeros((1, 4))
    res = {
        "Layer PSNR": float(np.mean(all_layer_psnr)),
        "Layer SSIM": float(np.mean(all_layer_ssim)),
        "Layer FID": fid_layers,
        "Mask IoU": float(np.mean(mask_arr[:, 0])),
        "Mask Precision": float(np.mean(mask_arr[:, 1])),
        "Mask Recall": float(np.mean(mask_arr[:, 2])),
        "Mask F1": float(np.mean(mask_arr[:, 3])),
        "Composite PSNR": float(np.mean(all_comp_psnr)),
        "Composite SSIM": float(np.mean(all_comp_ssim)),
        "Composite FID": fid_comp,
        "Num Samples": len(pred_samples),
        "Num Layers Evaluated": len(all_layer_psnr),
    }
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=64)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    args = parser.parse_args()

    results = evaluate(args.gt_dir, args.pred_dir, args.output_dir, args.num_processes, args.fid_batch_size)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for k, v in results.items():
        print(f"{k}: {v}")
