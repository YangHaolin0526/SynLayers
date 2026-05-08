import os
import json
import logging
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def rgba_to_rgb_masked(rgba_img: np.ndarray) -> np.ndarray:
    rgba = rgba_img.astype(np.float32)
    if rgba.max() <= 1.0:
        rgba = rgba * 255.0

    if rgba.shape[-1] == 4:
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:] / 255.0
        background = np.ones_like(rgb) * 128.0
        out = rgb * alpha + background * (1 - alpha)
        return out.astype(rgba_img.dtype)

    return rgba[..., :3].astype(rgba_img.dtype)


def load_gt_composite(gt_dir: str, sample_name: str):
    gt_path = os.path.join(gt_dir, f"{sample_name}.png")
    if not os.path.exists(gt_path):
        return None
    return Image.open(gt_path).convert("RGBA")


def load_pred_composite(pred_dir: str, sample_name: str):
    pred_path = os.path.join(pred_dir, sample_name, "merged.png")
    if not os.path.exists(pred_path):
        return None
    return Image.open(pred_path).convert("RGBA")


def process_case(args):
    sample_name, gt_dir, pred_dir = args

    gt_img = load_gt_composite(gt_dir, sample_name)
    pred_img = load_pred_composite(pred_dir, sample_name)
    if gt_img is None or pred_img is None:
        return None

    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img)

    if gt_np.size == 0 or pred_np.size == 0:
        return None

    if gt_np.shape != pred_np.shape:
        h = min(gt_np.shape[0], pred_np.shape[0])
        w = min(gt_np.shape[1], pred_np.shape[1])
        gt_np = gt_np[:h, :w]
        pred_np = pred_np[:h, :w]

    gt_rgb = rgba_to_rgb_masked(gt_np)
    pred_rgb = rgba_to_rgb_masked(pred_np)

    case_psnr = psnr(gt_rgb, pred_rgb, data_range=255)
    case_ssim = ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255)

    return case_psnr, case_ssim, gt_rgb, pred_rgb


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
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    dataset = ImageListDataset(img_list, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    feats = []
    for batch in tqdm(loader, desc="  Inception features (pytorch-fid)", leave=False):
        pred = model(batch.to(device))[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
        feats.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(feats1, feats2, eps=1e-6):
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def evaluate(gt_dir, pred_dir, output_dir, num_processes=16, fid_batch_size=64):
    del output_dir  # kept for CLI symmetry with other evaluators

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_samples = sorted(
        d
        for d in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, d))
        and os.path.exists(os.path.join(pred_dir, d, "merged.png"))
    )

    logger.info(
        "Found %d predicted samples. Starting composite-only evaluation with %d workers...",
        len(pred_samples),
        num_processes,
    )

    all_comp_psnr, all_comp_ssim = [], []
    gt_comp_for_fid, pred_comp_for_fid = [], []

    tasks = [(sample_name, gt_dir, pred_dir) for sample_name in pred_samples]
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks), desc="Processing samples")
        )

    valid_samples = 0
    for result in results:
        if result is None:
            continue
        case_psnr, case_ssim, gt_comp_rgb, pred_comp_rgb = result
        all_comp_psnr.append(case_psnr)
        all_comp_ssim.append(case_ssim)
        gt_comp_for_fid.append(gt_comp_rgb)
        pred_comp_for_fid.append(pred_comp_rgb)
        valid_samples += 1

    if not gt_comp_for_fid:
        raise ValueError("No valid sample pairs were found for evaluation.")

    logger.info("Loading Inception v3 (pytorch-fid weights, 2048-dim features)...")
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device).eval()

    fid_comp = (
        calculate_fid(
            get_inception_features(gt_comp_for_fid, inception, device, fid_batch_size),
            get_inception_features(pred_comp_for_fid, inception, device, fid_batch_size),
        )
        if len(gt_comp_for_fid) > 1
        else float("nan")
    )

    return {
        "Composite PSNR": float(np.mean(all_comp_psnr)),
        "Composite SSIM": float(np.mean(all_comp_ssim)),
        "Composite FID": fid_comp,
        "Num Samples": valid_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    args = parser.parse_args()

    results = evaluate(args.gt_dir, args.pred_dir, args.output_dir, args.num_processes, args.fid_batch_size)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for k, v in results.items():
        print(f"{k}: {v}")
