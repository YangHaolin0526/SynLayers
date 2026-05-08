import argparse
import glob
import json
import logging
import os
import random
import subprocess
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tools.dataset import BackgroundDataset, BackgroundIterableDataset


logger = logging.getLogger(__name__)


def iter_samples(dataset, streaming):
    if streaming:
        for sample in dataset:
            yield sample
    else:
        for idx in range(len(dataset)):
            yield dataset[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample background images for SynLayers.")
    parser.add_argument("--dataset-name", default="laion/laion2B-en-aesthetic")
    parser.add_argument(
        "--data-files",
        default="/project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image/*.parquet",
        help="Parquet glob or list file.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--url-column", default="URL")
    parser.add_argument("--text-column", default="TEXT")
    parser.add_argument("--hash-column", default="hash")
    parser.add_argument(
        "--image-root",
        default="/project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image",
        help="Local directory with downloaded images named by hash.",
    )
    parser.add_argument(
        "--image-extensions",
        default=".jpg,.png,.jpeg,.webp",
        help="Comma-separated extensions to try for local images.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-dir", default="./outputs/backgrounds")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save images if found in image-root.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download a subset into image-root using img2dataset.",
    )
    parser.add_argument(
        "--download-mode",
        choices=["auto", "img2dataset", "embedded"],
        default="auto",
        help="Download mode: auto-detect URL vs embedded bytes.",
    )
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--build-splits", action="store_true")
    parser.add_argument("--train-count", type=int, default=19000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading/extracting images that already exist in image-root.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Log progress every N extracted images.",
    )
    parser.add_argument(
        "--embedded-image-column",
        default="whole_image",
        help="Struct column containing embedded image bytes.",
    )
    parser.add_argument(
        "--embedded-image-columns",
        default=None,
        help="Comma-separated embedded image columns to try in order.",
    )
    parser.add_argument(
        "--embedded-image-bytes-key",
        default="bytes",
        help="Key inside embedded image struct that stores raw bytes.",
    )
    parser.add_argument(
        "--embedded-image-path-key",
        default="path",
        help="Key inside embedded image struct that stores a path (if any).",
    )
    parser.add_argument(
        "--embedded-caption-column",
        default="whole_caption",
        help="Caption column for embedded images.",
    )
    parser.add_argument(
        "--embedded-id-column",
        default="id",
        help="ID column for embedded images.",
    )
    parser.add_argument(
        "--size-multiple",
        type=int,
        default=8,
        help="Round width/height up to a multiple of this value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use dataset order instead of random sampling when building splits.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow writing splits even if there are not enough images.",
    )
    parser.add_argument(
        "--id-as-path",
        action="store_true",
        help="Store image path in the id field instead of the raw key.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    image_extensions = [ext.strip() for ext in args.image_extensions.split(",") if ext.strip()]

    if args.download:
        parquet_files = _expand_parquet_files(args.data_files)
        if not parquet_files:
            raise ValueError("No parquet files found. Check --data-files.")
        os.makedirs(args.image_root, exist_ok=True)
        download_mode = args.download_mode
        if args.embedded_image_columns:
            embedded_image_columns = [
                col.strip() for col in args.embedded_image_columns.split(",") if col.strip()
            ]
        else:
            embedded_image_columns = [args.embedded_image_column]
        if download_mode == "auto":
            if _parquet_has_column(parquet_files, args.url_column):
                download_mode = "img2dataset"
            elif any(
                _parquet_has_column(parquet_files, col) for col in embedded_image_columns
            ):
                download_mode = "embedded"
            else:
                raise ValueError(
                    "Could not detect download mode: missing URL and embedded image columns."
                )
        if download_mode == "img2dataset":
            url_list_path = _prepare_download_parquet(
                parquet_files=parquet_files,
                output_dir=args.output_dir,
                count=args.count,
                seed=args.seed,
                url_column=args.url_column,
                text_column=args.text_column,
                hash_column=args.hash_column,
            )
            cmd = [
                "img2dataset",
                "--url_list",
                url_list_path,
                "--input_format",
                "parquet",
                "--url_col",
                args.url_column,
                "--caption_col",
                args.text_column,
                "--output_format",
                "files",
                "--output_folder",
                args.image_root,
                "--processes_count",
                str(args.processes),
                "--thread_count",
                str(args.threads),
                "--image_size",
                str(args.resize),
                "--resize_mode",
                "keep_ratio",
            ]
            logger.info("Downloading %d images into %s", args.count, args.image_root)
            subprocess.run(cmd, check=True)
        else:
            logger.info(
                "Extracting %d embedded images into %s",
                args.count,
                args.image_root,
            )
            download_embedded_images(
                parquet_files=parquet_files,
                image_root=args.image_root,
                output_dir=args.output_dir,
                count=args.count,
                seed=args.seed,
                sequential=args.sequential,
                id_column=args.embedded_id_column,
                caption_column=args.embedded_caption_column,
                image_columns=embedded_image_columns,
                image_bytes_key=args.embedded_image_bytes_key,
                image_path_key=args.embedded_image_path_key,
                image_extensions=image_extensions,
                skip_existing=args.skip_existing,
                progress_interval=args.progress_interval,
            )

    if args.build_splits:
        if _has_img2dataset_parquet(args.image_root):
            build_splits_from_img2dataset(
                image_root=args.image_root,
                output_dir=args.output_dir,
                train_count=args.train_count,
                val_count=args.val_count,
                test_count=args.test_count,
                seed=args.seed,
                sequential=args.sequential,
                allow_partial=args.allow_partial,
                id_as_path=args.id_as_path,
                image_extensions=image_extensions,
                size_multiple=args.size_multiple,
            )
        else:
            build_splits(
                data_files=args.data_files,
                image_root=args.image_root,
                image_extensions=image_extensions,
                output_dir=args.output_dir,
                train_count=args.train_count,
                val_count=args.val_count,
                test_count=args.test_count,
                seed=args.seed,
                url_column=args.url_column,
                text_column=args.text_column,
                hash_column=args.hash_column,
                sequential=args.sequential,
                allow_partial=args.allow_partial,
                size_multiple=args.size_multiple,
            )
        return

    if args.streaming:
        dataset = BackgroundIterableDataset(
            dataset_name=args.dataset_name,
            data_files=args.data_files,
            split=args.split,
            cache_dir=args.cache_dir,
            url_column=args.url_column,
            text_column=args.text_column,
            hash_column=args.hash_column,
            image_root=args.image_root,
            image_extensions=image_extensions,
            image_size=args.image_size,
            require_image=args.save_images,
        )
    else:
        dataset = BackgroundDataset(
            dataset_name=args.dataset_name,
            data_files=args.data_files,
            split=args.split,
            cache_dir=args.cache_dir,
            url_column=args.url_column,
            text_column=args.text_column,
            hash_column=args.hash_column,
            image_root=args.image_root,
            image_extensions=image_extensions,
            image_size=args.image_size,
            max_items=args.count * 5,
            require_image=args.save_images,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    captions_path = os.path.join(args.output_dir, "captions.jsonl")

    saved = 0
    with open(captions_path, "w", encoding="utf-8") as captions_file:
        for sample in iter_samples(dataset, args.streaming):
            image = sample.get("image")
            filename = None
            if args.save_images:
                if image is None:
                    logger.warning("Skipping sample: local image not found.")
                    continue
                filename = f"background_{saved:03d}.png"
                image.save(os.path.join(args.output_dir, filename))
            captions_file.write(
                json.dumps(
                    {
                        "file": filename,
                        "url": sample.get("url"),
                        "text": sample.get("text"),
                        "width": sample.get("width"),
                        "height": sample.get("height"),
                        "hash": sample.get("hash"),
                        "aesthetic": sample.get("aesthetic"),
                        "punsafe": sample.get("punsafe"),
                        "pwatermark": sample.get("pwatermark"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            saved += 1
            if saved >= args.count:
                break

    logger.info("Saved %d backgrounds to %s", saved, args.output_dir)


def _expand_parquet_files(data_files):
    if isinstance(data_files, (list, tuple)):
        return list(data_files)
    if not data_files:
        return []
    if os.path.exists(data_files) and data_files.endswith(".parquet"):
        return [data_files]
    return sorted(glob.glob(data_files))


def _parquet_has_column(parquet_files, column_name):
    if not column_name:
        return False
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        if column_name in parquet_file.schema.names:
            return True
        schema_arrow = getattr(parquet_file, "schema_arrow", None)
        if schema_arrow is not None and column_name in schema_arrow.names:
            return True
    return False


def _has_img2dataset_parquet(image_root):
    if not image_root or not os.path.exists(image_root):
        return False
    return bool(glob.glob(os.path.join(image_root, "*.parquet")))


def _prepare_download_parquet(
    parquet_files,
    output_dir,
    count,
    seed,
    url_column,
    text_column,
    hash_column,
):
    os.makedirs(output_dir, exist_ok=True)
    if len(parquet_files) == 1:
        return parquet_files[0]
    rng = random.Random(seed)
    columns = [
        url_column,
        text_column,
        hash_column,
        "WIDTH",
        "HEIGHT",
        "aesthetic",
        "punsafe",
        "pwatermark",
    ]
    sampled = _reservoir_sample_parquet(
        parquet_files=parquet_files,
        target_count=count,
        rng=rng,
        columns=columns,
    )
    if not sampled:
        raise ValueError("Failed to sample rows from parquet files.")
    table = pa.Table.from_pylist(sampled)
    out_path = os.path.join(output_dir, "laion_download_sample.parquet")
    pq.write_table(table, out_path)
    logger.info("Wrote sampled parquet list to %s", out_path)
    return out_path


def _detect_image_extension(image):
    fmt = (image.format or "").upper()
    if fmt == "JPEG":
        return "jpg"
    if fmt == "PNG":
        return "png"
    if fmt == "WEBP":
        return "webp"
    return "jpg"


def _collect_existing_images(image_root, image_extensions):
    if not image_root or not os.path.exists(image_root):
        return {}
    image_map = {}
    for root, _, files in os.walk(image_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in image_extensions:
                stem = os.path.splitext(name)[0]
                image_map[stem] = os.path.join(root, name)
    return image_map


def _save_image_bytes(image_bytes, output_path):
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            ext = _detect_image_extension(img)
            if ext == "jpg":
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            output_path = os.path.splitext(output_path)[0] + f".{ext}"
            img.save(output_path)
            return output_path, img.size
    except Exception as exc:
        logger.warning("Failed to decode image bytes: %s", exc)
        return None, None


def _iter_embedded_rows(
    parquet_files,
    id_column,
    caption_column,
    image_columns,
    image_bytes_key,
    image_path_key,
):
    columns = [id_column, caption_column] + list(image_columns)
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=256):
            batch_dict = batch.to_pydict()
            batch_len = len(batch)
            for i in range(batch_len):
                image_bytes = None
                image_path = None
                for image_column in image_columns:
                    image_struct = batch_dict.get(image_column, [None])[i] or {}
                    image_bytes = image_struct.get(image_bytes_key)
                    image_path = image_struct.get(image_path_key)
                    if image_bytes:
                        break
                if not image_bytes:
                    continue
                yield {
                    "id": batch_dict.get(id_column, [None])[i],
                    "caption": batch_dict.get(caption_column, [None])[i],
                    "bytes": image_bytes,
                    "path": image_path,
                }


def download_embedded_images(
    parquet_files,
    image_root,
    output_dir,
    count,
    seed,
    sequential,
    id_column,
    caption_column,
    image_columns,
    image_bytes_key,
    image_path_key,
    image_extensions,
    skip_existing,
    progress_interval,
):
    os.makedirs(image_root, exist_ok=True)
    rng = random.Random(seed)
    selected_ids = None
    if not sequential:
        sampled = _reservoir_sample_parquet(
            parquet_files=parquet_files,
            target_count=count,
            rng=rng,
            columns=[id_column],
        )
        selected_ids = {
            str(row.get(id_column))
            for row in sampled
            if row.get(id_column) is not None
        }
        if not selected_ids:
            raise ValueError("Failed to sample IDs from parquet files.")

    image_extensions = image_extensions or [".jpg", ".png", ".jpeg", ".webp"]
    existing_map = _collect_existing_images(image_root, image_extensions) if skip_existing else {}
    if existing_map and len(existing_map) >= count:
        logger.info(
            "Found %d existing images in %s (target=%d).",
            len(existing_map),
            image_root,
            count,
        )
    metadata_rows = []
    for row in _iter_embedded_rows(
        parquet_files=parquet_files,
        id_column=id_column,
        caption_column=caption_column,
        image_columns=image_columns,
        image_bytes_key=image_bytes_key,
        image_path_key=image_path_key,
    ):
        image_id = row.get("id")
        if image_id is None:
            continue
        image_id = str(image_id)
        if selected_ids is not None and image_id not in selected_ids:
            continue
        saved_path = None
        size = None
        if image_id in existing_map:
            saved_path = existing_map[image_id]
            size = _get_image_size(saved_path)
        if saved_path is None:
            shard_dir = image_id[:5] if len(image_id) >= 5 else image_id
            target_dir = os.path.join(image_root, shard_dir)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, image_id)
            saved_path, size = _save_image_bytes(row["bytes"], target_path)
            if not saved_path:
                continue
        width, height = size if size else (None, None)
        metadata_rows.append(
            {
                "key": image_id,
                "caption": row.get("caption"),
                "status": "success",
                "width": width,
                "height": height,
            }
        )
        if progress_interval and len(metadata_rows) % progress_interval == 0:
            logger.info("Extracted %d/%d images...", len(metadata_rows), count)
        if sequential and len(metadata_rows) >= count:
            break
        if selected_ids is not None and len(metadata_rows) >= len(selected_ids):
            break

    if not metadata_rows:
        raise ValueError("No embedded images were extracted.")
    meta_table = pa.Table.from_pylist(metadata_rows)
    meta_path = os.path.join(image_root, "embedded_metadata.parquet")
    pq.write_table(meta_table, meta_path)
    logger.info("Wrote embedded metadata to %s", meta_path)


def _reservoir_sample_parquet(parquet_files, target_count, rng, columns):
    sample = []
    total_seen = 0
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            batch_dict = batch.to_pydict()
            batch_len = len(batch)
            for i in range(batch_len):
                row = {col: batch_dict.get(col, [None])[i] for col in columns}
                total_seen += 1
                if len(sample) < target_count:
                    sample.append(row)
                else:
                    j = rng.randint(0, total_seen - 1)
                    if j < target_count:
                        sample[j] = row
    return sample


def _iter_img2dataset_rows(image_root):
    parquet_files = sorted(glob.glob(os.path.join(image_root, "*.parquet")))
    if not parquet_files:
        return
    columns = ["key", "caption", "status", "width", "height"]
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            batch_dict = batch.to_pydict()
            batch_len = len(batch)
            for i in range(batch_len):
                status = batch_dict.get("status", [None])[i]
                if status and status != "success":
                    continue
                key = batch_dict.get("key", [None])[i]
                caption = batch_dict.get("caption", [None])[i]
                width = batch_dict.get("width", [None])[i]
                height = batch_dict.get("height", [None])[i]
                if key is None:
                    continue
                key_str = str(key)
                yield {
                    "id": key_str,
                    "caption": caption,
                    "width": width,
                    "height": height,
                }


def _image_path_from_id(image_root, key_str, image_extensions):
    if not key_str:
        return None
    shard_dir = key_str[:5]
    for ext in image_extensions:
        path = os.path.join(image_root, shard_dir, f"{key_str}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(image_root, shard_dir, f"{key_str}.jpg")


def _round_up_multiple(value, multiple):
    if multiple <= 1:
        return int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def _get_image_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        logger.warning("Failed to read image size for %s: %s", path, exc)
        return None


def build_splits_from_img2dataset(
    image_root,
    output_dir,
    train_count,
    val_count,
    test_count,
    seed,
    sequential=False,
    allow_partial=False,
    id_as_path=False,
    image_extensions=None,
    size_multiple=8,
):
    os.makedirs(output_dir, exist_ok=True)
    total_needed = train_count + val_count + test_count
    image_extensions = image_extensions or [".jpg", ".png", ".jpeg", ".webp"]
    items = []
    if sequential:
        for row in _iter_img2dataset_rows(image_root):
            items.append(row)
            if len(items) >= total_needed:
                break
    else:
        rng = random.Random(seed)
        total_seen = 0
        for row in _iter_img2dataset_rows(image_root):
            total_seen += 1
            if len(items) < total_needed:
                items.append(row)
            else:
                j = rng.randint(0, total_seen - 1)
                if j < total_needed:
                    items[j] = row
        rng.shuffle(items)

    if len(items) < total_needed:
        if not allow_partial:
            raise ValueError(
                f"Only found {len(items)} matching images (needed {total_needed})."
            )
        logger.warning(
            "Only found %d matching images (needed %d).",
            len(items),
            total_needed,
        )

    if id_as_path:
        for item in items:
            item["id"] = _image_path_from_id(image_root, item["id"], image_extensions)

    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count : train_count + val_count + test_count]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                image_path = row.get("path")
                if not image_path:
                    image_id = row.get("id")
                    if image_id:
                        if os.path.isabs(image_id):
                            image_path = image_id
                        else:
                            image_path = _image_path_from_id(
                                image_root, image_id, image_extensions
                            )
                if image_path:
                    row["path"] = image_path
                    size = _get_image_size(image_path)
                    if size:
                        width, height = size
                    else:
                        width = row.get("width")
                        height = row.get("height")
                    if width and height:
                        row["width"] = _round_up_multiple(int(width), size_multiple)
                        row["height"] = _round_up_multiple(int(height), size_multiple)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(output_dir, "train.jsonl"), train_items)
    write_jsonl(os.path.join(output_dir, "val.jsonl"), val_items)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test_items)

    logger.info(
        "Wrote splits to %s (train=%d, val=%d, test=%d)",
        output_dir,
        len(train_items),
        len(val_items),
        len(test_items),
    )


def _scan_images(image_root, image_extensions):
    if not image_root or not os.path.exists(image_root):
        return {}
    image_map = {}
    for root, _, files in os.walk(image_root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in image_extensions:
                stem = os.path.splitext(name)[0]
                image_map[stem] = os.path.join(root, name)
    return image_map


def _collect_metadata(
    parquet_files,
    image_map,
    target_count,
    url_column,
    text_column,
    hash_column,
):
    selected = []
    hashes = set(image_map.keys())
    if not hashes:
        return selected
    columns = [
        hash_column,
        url_column,
        text_column,
        "WIDTH",
        "HEIGHT",
        "aesthetic",
        "punsafe",
        "pwatermark",
    ]
    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4096):
            batch_dict = batch.to_pydict()
            for i in range(len(batch)):
                hash_value = batch_dict.get(hash_column, [None])[i]
                if hash_value is None:
                    continue
                hash_str = str(hash_value)
                path = image_map.get(hash_str)
                if not path:
                    continue
                selected.append(
                    {
                        "file": path,
                        "url": batch_dict.get(url_column, [None])[i],
                        "text": batch_dict.get(text_column, [None])[i],
                        "width": batch_dict.get("WIDTH", [None])[i],
                        "height": batch_dict.get("HEIGHT", [None])[i],
                        "hash": hash_str,
                        "aesthetic": batch_dict.get("aesthetic", [None])[i],
                        "punsafe": batch_dict.get("punsafe", [None])[i],
                        "pwatermark": batch_dict.get("pwatermark", [None])[i],
                    }
                )
                if len(selected) >= target_count:
                    return selected
    return selected


def build_splits(
    data_files,
    image_root,
    image_extensions,
    output_dir,
    train_count,
    val_count,
    test_count,
    seed,
    url_column,
    text_column,
    hash_column,
    sequential=False,
    allow_partial=False,
    size_multiple=8,
):
    os.makedirs(output_dir, exist_ok=True)
    parquet_files = _expand_parquet_files(data_files)
    if not parquet_files:
        raise ValueError("No parquet files found. Check --data-files.")

    image_map = _scan_images(image_root, image_extensions)
    if not image_map:
        raise ValueError("No images found in image_root.")

    total_needed = train_count + val_count + test_count
    logger.info(
        "Collecting %d samples from %d parquet files (images=%d)",
        total_needed,
        len(parquet_files),
        len(image_map),
    )
    items = _collect_metadata(
        parquet_files=parquet_files,
        image_map=image_map,
        target_count=total_needed,
        url_column=url_column,
        text_column=text_column,
        hash_column=hash_column,
    )
    if len(items) < total_needed:
        if not allow_partial:
            raise ValueError(
                f"Only found {len(items)} matching images (needed {total_needed})."
            )
        logger.warning(
            "Only found %d matching images (needed %d).",
            len(items),
            total_needed,
        )

    if not sequential:
        rng = random.Random(seed)
        rng.shuffle(items)
    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count : train_count + val_count + test_count]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                image_path = row.get("path") or row.get("file")
                if image_path:
                    row["path"] = image_path
                    size = _get_image_size(image_path)
                    if size:
                        width, height = size
                    else:
                        width = row.get("width")
                        height = row.get("height")
                    if width and height:
                        row["width"] = _round_up_multiple(int(width), size_multiple)
                        row["height"] = _round_up_multiple(int(height), size_multiple)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(output_dir, "train.jsonl"), train_items)
    write_jsonl(os.path.join(output_dir, "val.jsonl"), val_items)
    write_jsonl(os.path.join(output_dir, "test.jsonl"), test_items)

    logger.info(
        "Wrote splits to %s (train=%d, val=%d, test=%d)",
        output_dir,
        len(train_items),
        len(val_items),
        len(test_items),
    )


if __name__ == "__main__":
    main()

'''
python -m tools.sample_backgrounds \
  --download \
  --count 20100 \
  --build-splits \
  --train-count 19000 \
  --val-count 1000 \
  --test-count 200 \
  --data-files "/project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image/*.parquet" \
  --image-root "/project/llmsvgen/share/data/kmw_layered_dataset/laion2B-en-aesthetic-image" \
  --output-dir "/project/llmsvgen/jinmin/SynLayers/data/laion2b_splits"

python -m tools.sample_backgrounds \
  --download \
  --build-splits \
  --count 40200 \
  --sequential \
  --id-as-path \
  --train-count 19000 \
  --val-count 1000 \
  --test-count 200 \
  --data-files "/project/llmsvgen/share/data/kmw_layered_dataset/PrismLayersPro-image/data/*.parquet" \
  --image-root "/project/llmsvgen/share/data/kmw_layered_dataset/PrismLayersPro-image/data/haolin/PrismLayersPro-image" \
  --output-dir "/project/llmsvgen/jinmin/SynLayers/data/prismlayerspro_splits"
'''