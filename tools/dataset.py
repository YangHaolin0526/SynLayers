import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
import torchvision.transforms as T
from collections import defaultdict


def collate_fn(batch):
    pixels_RGBA = [torch.stack(item["pixel_RGBA"]) for item in batch]  # [L, C, H, W]
    pixels_RGB  = [torch.stack(item["pixel_RGB"])  for item in batch]  # [L, C, H, W]
    pixels_RGBA = torch.stack(pixels_RGBA)  # [B, L, C, H, W]
    pixels_RGB  = torch.stack(pixels_RGB)   # [B, L, C, H, W]

    return {
        "pixel_RGBA": pixels_RGBA,
        "pixel_RGB": pixels_RGB,
        "whole_img": [item["whole_img"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": [item["height"] for item in batch],
        "width": [item["width"] for item in batch],
        "layout": [item["layout"] for item in batch],
    }

class LayoutTrainDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        full_dataset = load_dataset(
            "artplus/PrismLayersPro",
            cache_dir=data_dir,
        )
        full_dataset = concatenate_datasets(list(full_dataset.values()))

        if "style_category" not in full_dataset.column_names:
            raise ValueError("Dataset must contain a 'style_category' field to split by class.")

        categories = np.array(full_dataset["style_category"])
        category_to_indices = defaultdict(list)
        for i, cat in enumerate(categories):
            category_to_indices[cat].append(i)

        subsets = []
        for cat, indices in category_to_indices.items():
            total_len = len(indices)
            idx_90 = int(total_len * 0.9)
            idx_95 = int(total_len * 0.95)

            if split == "train":
                selected_idx = indices[:idx_90]
            elif split == "test":
                selected_idx = indices[idx_90:idx_95]
            elif split == "val":
                selected_idx = indices[idx_95:]
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")

            subsets.append(full_dataset.select(selected_idx))

        self.dataset = concatenate_datasets(subsets)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        def rgba2rgb(img_RGBA):
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x):
            if isinstance(x, str):
                img_RGBA = Image.open(x).convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            else:
                img_RGBA = x.convert("RGBA")
                img_RGB = rgba2rgb(img_RGBA)
            return img_RGBA, img_RGB

        whole_img_RGBA, whole_img_RGB = get_img(item["whole_image"])
        whole_cap = item["whole_caption"]
        W, H = whole_img_RGBA.size
        base_layout = [0, 0, W, H]  # xyxy with exclusive end coordinates

        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB  = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        base_img_RGBA, base_img_RGB = get_img(item["base_image"])
        layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)

        layer_count = item["layer_count"]
        for i in range(layer_count):
            key = f"layer_{i:02d}"
            img_RGBA, img_RGB = get_img(item[key])
            
            w0, h0, w1, h1 = item[f"{key}_box"]

            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            W_img, H_img = w1 - w0, h1 - h0
            if img_RGBA.size != (W_img, H_img):
                img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                img_RGB  = img_RGB.resize((W_img, H_img), Image.BILINEAR)

            canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
            canvas_RGB.paste(img_RGB, (w0, h0))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([w0, h0, w1, h1])

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }


class LayoutDatasetFixedSplit(Dataset):
    """
    HuggingFace PrismLayersPro with a fixed index-based split.
    Total 20,000 samples: train = [0, 19500), test = [19500, 20000).

    For test split, use start_index and max_samples to select a sub-range:
      start_index=200, max_samples=100  ->  samples 019700-019799
      start_index=0,   max_samples=100  ->  samples 019500-019599
    """

    TRAIN_END = 19500
    TOTAL = 20000

    def __init__(self, data_dir, split="train", start_index=0, max_samples=None):
        full_dataset = load_dataset(
            "artplus/PrismLayersPro",
            cache_dir=data_dir,
        )
        full_dataset = concatenate_datasets(list(full_dataset.values()))

        if split == "train":
            self.dataset = full_dataset.select(range(self.TRAIN_END))
            self.global_offset = 0
        elif split == "test":
            self.dataset = full_dataset.select(range(self.TRAIN_END, self.TOTAL))
            self.global_offset = self.TRAIN_END
        else:
            raise ValueError("split must be 'train' or 'test'")

        end_index = len(self.dataset)
        if max_samples is not None:
            end_index = min(start_index + max_samples, len(self.dataset))
        self.dataset = self.dataset.select(range(start_index, end_index))
        self.global_offset += start_index

        self.to_tensor = T.ToTensor()
        print(f"[INFO] LayoutDatasetFixedSplit: split={split}, "
              f"global range=[{self.global_offset}, {self.global_offset + len(self.dataset)}), "
              f"samples={len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        def rgba2rgb(img_RGBA):
            img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
            img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
            return img_RGB

        def get_img(x):
            if isinstance(x, str):
                img_RGBA = Image.open(x).convert("RGBA")
            else:
                img_RGBA = x.convert("RGBA")
            return img_RGBA, rgba2rgb(img_RGBA)

        whole_img_RGBA, whole_img_RGB = get_img(item["whole_image"])
        whole_cap = item["whole_caption"]
        W, H = whole_img_RGBA.size
        base_layout = [0, 0, W, H]

        layer_image_RGBA = [self.to_tensor(whole_img_RGBA)]
        layer_image_RGB = [self.to_tensor(whole_img_RGB)]
        layout = [base_layout]

        base_img_RGBA, base_img_RGB = get_img(item["base_image"])
        layer_image_RGBA.append(self.to_tensor(base_img_RGBA))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)

        layer_count = item["layer_count"]
        for i in range(layer_count):
            key = f"layer_{i:02d}"
            img_RGBA, img_RGB = get_img(item[key])

            w0, h0, w1, h1 = item[f"{key}_box"]

            canvas_RGBA = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas_RGB = Image.new("RGB", (W, H), (128, 128, 128))

            W_img, H_img = w1 - w0, h1 - h0
            if img_RGBA.size != (W_img, H_img):
                img_RGBA = img_RGBA.resize((W_img, H_img), Image.BILINEAR)
                img_RGB = img_RGB.resize((W_img, H_img), Image.BILINEAR)

            canvas_RGBA.paste(img_RGBA, (w0, h0), img_RGBA)
            canvas_RGB.paste(img_RGB, (w0, h0))

            layer_image_RGBA.append(self.to_tensor(canvas_RGBA))
            layer_image_RGB.append(self.to_tensor(canvas_RGB))
            layout.append([w0, h0, w1, h1])

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": whole_cap,
            "height": H,
            "width": W,
            "layout": layout,
        }


def prism_collate_fn(batch):
    """Collate function for PrismBlendDataset."""
    pixels_RGBA = [torch.stack(item["pixel_RGBA"]) for item in batch]
    pixels_RGB = [torch.stack(item["pixel_RGB"]) for item in batch]
    pixels_RGBA = torch.stack(pixels_RGBA)
    pixels_RGB = torch.stack(pixels_RGB)

    return {
        "pixel_RGBA": pixels_RGBA,
        "pixel_RGB": pixels_RGB,
        "whole_img": [item["whole_img"] for item in batch],
        "caption": [item["caption"] for item in batch],
        "height": [item["height"] for item in batch],
        "width": [item["width"] for item in batch],
        "layout": [item["layout"] for item in batch],
    }


class PrismBlendDataset(Dataset):
    """
    Dataset for PrismLayersPro blended data.
    
    Loads from local directory structure (following PrismLayersPro convention):
    - data_dir/sample_XXXXXX/metadata.json
    - data_dir/sample_XXXXXX/whole_image.png
    - data_dir/sample_XXXXXX/base_image.png
    - data_dir/sample_XXXXXX/layer_00.png, layer_01.png, ...
    
    Boxes are in xyxy format: [x0, y0, x1, y1]
    All layer images have transparent backgrounds.
    """
    
    def __init__(self, data_dir: str, jsonl_path: str = None, target_size: int = 512, split: str = "all", max_layer_num: int = None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.max_layer_num = max_layer_num
        self.to_tensor = T.ToTensor()
        
        # Load samples
        if jsonl_path and os.path.exists(jsonl_path):
            self.samples = self._load_from_jsonl(jsonl_path)
        else:
            self.samples = self._load_from_directory(data_dir)
        
        # Filter samples exceeding max_layer_num (if specified)
        # Total layers = 2 (whole_image + base_image) + layer_count
        if max_layer_num is not None:
            original_count = len(self.samples)
            self.samples = [
                s for s in self.samples
                if (2 + s.get('layer_count', 0)) <= max_layer_num
            ]
            filtered_count = original_count - len(self.samples)
            if filtered_count > 0:
                print(f"[INFO] Filtered {filtered_count} samples exceeding max_layer_num={max_layer_num}")
        
        # Split dataset (only if explicitly requested, default is "all" = use all samples)
        # Usually you have separate train/test datasets, so no splitting needed
        if split == "train_split":
            self.samples = self.samples[:int(len(self.samples) * 0.9)]
        elif split == "test_split":
            self.samples = self.samples[int(len(self.samples) * 0.9):int(len(self.samples) * 0.95)]
        elif split == "val_split":
            self.samples = self.samples[int(len(self.samples) * 0.95):]
        # "all", "train", "test" -> use all samples from the provided jsonl/directory
    
    def _load_from_jsonl(self, jsonl_path: str):
        """Load samples from JSONL file."""
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def _load_from_directory(self, data_dir: str):
        """Load samples from directory structure."""
        samples = []
        for name in sorted(os.listdir(data_dir)):
            sample_dir = os.path.join(data_dir, name)
            if os.path.isdir(sample_dir) and name.startswith('sample_'):
                metadata_path = os.path.join(sample_dir, 'metadata.json')
                #metadata_path = os.path.join(sample_dir, 'metadata_old.json') # old for original_1024.
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        samples.append(json.load(f))
        return samples

    def __len__(self):
        return len(self.samples)

    def _rgba2rgb(self, img_RGBA):
        """Convert RGBA to RGB with gray background."""
        img_RGB = Image.new("RGB", img_RGBA.size, (128, 128, 128))
        img_RGB.paste(img_RGBA, mask=img_RGBA.split()[3])
        return img_RGB
    
    def _get_sample_dir(self, sample):
        """Get the directory for a sample."""
        # Try sample_dir first
        sample_dir = sample.get('sample_dir', '')
        if sample_dir:
            full_path = os.path.join(self.data_dir, sample_dir)
            if os.path.exists(full_path):
                return full_path
        
        return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self._get_sample_dir(sample)
        
        if not sample_dir:
            raise ValueError(f"Could not find sample directory for index {idx}")
        
        source_size = sample.get('width', self.target_size)
        caption = sample.get('whole_caption', '')
        
        # Scale factor (source -> target)
        scale = self.target_size / source_size
        
        # Load whole_image (composite)
        whole_img_path = os.path.join(sample_dir, 'whole_image.png')
        if os.path.exists(whole_img_path):
            whole_img = Image.open(whole_img_path).convert('RGBA')
        else:
            whole_img = Image.new('RGBA', (source_size, source_size), (128, 128, 128, 255))
        
        # Resize if needed
        if whole_img.size != (self.target_size, self.target_size):
            whole_img = whole_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        whole_img_RGB = self._rgba2rgb(whole_img)
        
        # Initialize layer lists with whole_image first
        layer_image_RGBA = [self.to_tensor(whole_img)]
        layer_image_RGB = [self.to_tensor(whole_img_RGB)]
        
        # Base layout (whole image) in xyxy format [x0, y0, x1, y1]
        W, H = self.target_size, self.target_size
        base_layout = [0, 0, W, H]  # xyxy with exclusive end coordinates
        layout = [base_layout]
        
        # Load base_image (background) as second layer
        base_img_path = os.path.join(sample_dir, 'base_image.png')
        if os.path.exists(base_img_path):
            base_img = Image.open(base_img_path).convert('RGBA')
            if base_img.size != (self.target_size, self.target_size):
                base_img = base_img.resize((self.target_size, self.target_size), Image.LANCZOS)
        else:
            base_img = Image.new('RGBA', (self.target_size, self.target_size), (0, 0, 0, 0))
        
        base_img_RGB = self._rgba2rgb(base_img)
        layer_image_RGBA.append(self.to_tensor(base_img))
        layer_image_RGB.append(self.to_tensor(base_img_RGB))
        layout.append(base_layout)  # background covers whole image
        
        # Load layers from metadata
        layers = sample.get('layers', [])
        
        for layer_info in layers:
            image_path = layer_info.get('image_path', '')
            box = layer_info.get('box', [0, 0, source_size, source_size])
            
            # Scale box (xyxy format)
            x0, y0, x1, y1 = box
            scaled_box = [
                int(x0 * scale),
                int(y0 * scale),
                int(x1 * scale),
                int(y1 * scale)
            ]
            
            # Load layer image
            # Handles two formats:
            #   1. Full-canvas (target_size x target_size) — use as-is
            #   2. Cropped (smaller than canvas) — place at bbox position on transparent canvas
            layer_path = os.path.join(sample_dir, image_path)
            if os.path.exists(layer_path):
                layer_img = Image.open(layer_path).convert('RGBA')
                if layer_img.size == (self.target_size, self.target_size):
                    # Already full-canvas, use directly
                    pass
                elif layer_img.size == (source_size, source_size) and source_size != self.target_size:
                    # Full-canvas at source resolution, just resize
                    layer_img = layer_img.resize((self.target_size, self.target_size), Image.LANCZOS)
                else:
                    # Cropped layer — resize to fit the scaled bbox and place on canvas
                    bw = max(1, scaled_box[2] - scaled_box[0])
                    bh = max(1, scaled_box[3] - scaled_box[1])
                    layer_resized = layer_img.resize((bw, bh), Image.LANCZOS)
                    layer_img = Image.new('RGBA', (self.target_size, self.target_size), (0, 0, 0, 0))
                    layer_img.paste(layer_resized, (scaled_box[0], scaled_box[1]), layer_resized)
            else:
                layer_img = Image.new('RGBA', (self.target_size, self.target_size), (0, 0, 0, 0))
            
            layer_img_RGB = self._rgba2rgb(layer_img)
            
            layer_image_RGBA.append(self.to_tensor(layer_img))
            layer_image_RGB.append(self.to_tensor(layer_img_RGB))
            layout.append(scaled_box)

        return {
            "pixel_RGBA": layer_image_RGBA,
            "pixel_RGB": layer_image_RGB,
            "whole_img": whole_img_RGB,
            "caption": caption,
            "height": H,
            "width": W,
            "layout": layout,
        }