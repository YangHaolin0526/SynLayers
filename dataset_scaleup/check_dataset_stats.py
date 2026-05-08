import os, json, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/project/llmsvgen/share/data/kmw_layered_dataset/PrismLayersPro-scaledup-1024-alpha-100k"
NUM_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 64


def process_sample(sample_name):
    meta_path = os.path.join(ROOT, sample_name, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(ROOT, sample_name, "metadata_old.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, 'r') as f:
        data = json.load(f)

    layers = data.get("layers", [])
    source_counts = Counter()
    type_counts = Counter()
    flags = {"alphavae": False, "laion": False, "caption": False, "prism": False}

    for layer in layers:
        source = layer.get("source", "unknown")
        layer_type = layer.get("type", None)
        source_counts[source] += 1

        if layer_type == "alphavae":
            flags["alphavae"] = True
            type_counts["alphavae"] += 1
        elif layer_type == "laion_foreground":
            flags["laion"] = True
            type_counts["laion_foreground"] += 1
        elif layer_type == "caption":
            flags["caption"] = True
            type_counts["caption_text"] += 1
        elif source == "base":
            flags["prism"] = True
            type_counts["prism_base"] += 1
        elif source == "donor":
            flags["prism"] = True
            type_counts["prism_donor"] += 1
        else:
            type_counts["other_" + source] += 1

    return {"n_layers": len(layers), "source_counts": source_counts, "type_counts": type_counts, "flags": flags}


if __name__ == "__main__":
    sample_dirs = sorted([d for d in os.listdir(ROOT) if d.startswith("sample_") and os.path.isdir(os.path.join(ROOT, d))])
    print(f"Dataset: {ROOT}")
    print(f"Total sample directories: {len(sample_dirs)}")
    print(f"Processing with {NUM_WORKERS} workers...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_sample, sample_dirs)

    results = [r for r in results if r is not None]
    n = len(results)
    print(f"Samples analyzed: {n}")

    layer_counts = [r["n_layers"] for r in results]
    total_layers = sum(layer_counts)
    total_source = Counter()
    total_type = Counter()
    has_alphavae = sum(1 for r in results if r["flags"]["alphavae"])
    has_laion = sum(1 for r in results if r["flags"]["laion"])
    has_caption = sum(1 for r in results if r["flags"]["caption"])
    has_prism = sum(1 for r in results if r["flags"]["prism"])

    for r in results:
        total_source.update(r["source_counts"])
        total_type.update(r["type_counts"])

    print(f"\n{'='*60}")
    print(f"LAYER COUNT STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean layers/sample: {np.mean(layer_counts):.2f}")
    print(f"  Median: {np.median(layer_counts):.0f}")
    print(f"  Min: {min(layer_counts)}, Max: {max(layer_counts)}")
    print(f"  Total layers: {total_layers}")

    print(f"\n{'='*60}")
    print(f"LAYER SOURCE FIELD (raw)")
    print(f"{'='*60}")
    for src, cnt in total_source.most_common():
        print(f"  {src:20s}: {cnt:8d} ({cnt/total_layers*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"LAYER TYPE (classified)")
    print(f"{'='*60}")
    for t, cnt in total_type.most_common():
        print(f"  {t:20s}: {cnt:8d} ({cnt/total_layers*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"SAMPLE-LEVEL PRESENCE (out of {n} samples)")
    print(f"{'='*60}")
    print(f"  Prism (base/donor):  {has_prism:7d} ({has_prism/n*100:.1f}%)")
    print(f"  AlphaVAE foreground: {has_alphavae:7d} ({has_alphavae/n*100:.1f}%)")
    print(f"  LAION background:    {has_laion:7d} ({has_laion/n*100:.1f}%)")
    print(f"  Caption/Text:        {has_caption:7d} ({has_caption/n*100:.1f}%)")
