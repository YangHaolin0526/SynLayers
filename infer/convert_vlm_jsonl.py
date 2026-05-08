import argparse
import json


def convert(input_path: str, output_path: str, canvas_size: int = 1024):
    converted_count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            vlm = json.loads(line)

            bboxes = vlm.get("bboxes", [])
            layers = []
            for i, bbox in enumerate(bboxes):
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                layers.append({
                    "layer_idx": i,
                    "box": [x0, y0, x1, y1],
                    "width_dst": x1 - x0,
                    "height_dst": y1 - y0,
                })

            record = {
                "sample_dir": vlm.get("sample_or_stem", ""),
                "whole_caption": vlm.get("whole_caption", ""),
                "layer_count": len(layers),
                "width": canvas_size,
                "height": canvas_size,
                "layers": layers,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"Converted {converted_count} samples: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VLM JSONL to prism_infer format")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--canvas_size", type=int, default=1024)
    args = parser.parse_args()
    convert(args.input, args.output, args.canvas_size)
