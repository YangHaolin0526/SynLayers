import argparse
import json
import os
import random
import re


FALLBACK_WORDS = [
    "sunset",
    "river",
    "forest",
    "city",
    "mountain",
    "garden",
    "morning",
    "night",
    "quiet",
    "dream",
    "light",
    "shadow",
    "breeze",
    "golden",
    "silver",
]


def extract_words(text):
    tokens = re.findall(r"[A-Za-z]+", text or "")
    return [t.lower() for t in tokens if len(t) > 1]


def choose_caption(words, rng):
    if len(words) < 3:
        words = FALLBACK_WORDS
    count = rng.randint(3, 5)
    if len(words) < count:
        return " ".join(rng.choices(words, k=count))
    return " ".join(rng.sample(words, count))


def random_box(width, height, rng):
    box_w = int(width * rng.uniform(0.48, 0.72))
    box_h = int(height * rng.uniform(0.18, 0.32))
    box_w = max(16, min(box_w, width))
    box_h = max(16, min(box_h, height))
    x = rng.randint(0, max(0, width - box_w))
    y = rng.randint(0, max(0, height - box_h))
    return x, y, box_w, box_h


def load_items(jsonl_path):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_entries(items, count, seed):
    rng = random.Random(seed)
    entries = []
    for i in range(count):
        item = items[i % len(items)]
        width = int(item.get("width", 512))
        height = int(item.get("height", 512))
        words = extract_words(item.get("caption", ""))
        caption = choose_caption(words, rng)
        x, y, box_w, box_h = random_box(width, height, rng)
        entries.append(
            {
                "path": item.get("path"),
                "caption": caption,
                "x": x,
                "y": y,
                "box_w": box_w,
                "box_h": box_h,
                "width": width,
                "height": height,
            }
        )
    return entries


def write_jsonl(entries, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate caption placement JSONL.")
    parser.add_argument(
        "--input-jsonl",
        default="/project/llmsvgen/jinmin/SynLayers/data/laion2b_splits/train.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/project/llmsvgen/jinmin/SynLayers/caption/captions.jsonl",
    )
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_items(args.input_jsonl)
    if not items:
        raise ValueError(f"No items found in {args.input_jsonl}")
    entries = build_entries(items, args.count, args.seed)
    write_jsonl(entries, args.output_jsonl)


if __name__ == "__main__":
    main()

