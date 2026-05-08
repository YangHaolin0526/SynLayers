import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.dataset import round_up_multiple


BASE_PALETTE = [
    (220, 30, 60),    # red
    (0, 180, 80),     # green
    (255, 90, 0),     # orange
    (255, 165, 0),    # amber
    (255, 80, 120),   # pink
    (180, 40, 200),   # purple
    (120, 0, 180),    # deep purple
    (30, 90, 220),    # blue
    (0, 160, 220),    # cyan
    (0, 0, 0),        # black
    (255, 255, 255),  # white (rare)
]


def jitter_color(rgb, rng, jitter=35):
    r, g, b = rgb
    r = max(0, min(255, r + rng.randint(-jitter, jitter)))
    g = max(0, min(255, g + rng.randint(-jitter, jitter)))
    b = max(0, min(255, b + rng.randint(-jitter, jitter)))
    return (r, g, b, 255)


def random_color_pair(rng):
    weighted = (
        BASE_PALETTE[:-1]
        + BASE_PALETTE[:-1]
        + [BASE_PALETTE[-1]]
    )
    left = jitter_color(rng.choice(weighted), rng, jitter=rng.randint(10, 40))
    right = jitter_color(rng.choice(weighted), rng, jitter=rng.randint(10, 40))
    return left, right


def make_horizontal_gradient(width, height, left_rgba, right_rgba):
    gradient = Image.new("RGBA", (width, height), left_rgba)
    if width <= 1:
        return gradient
    for x in range(width):
        t = x / (width - 1)
        r = int(left_rgba[0] * (1 - t) + right_rgba[0] * t)
        g = int(left_rgba[1] * (1 - t) + right_rgba[1] * t)
        b = int(left_rgba[2] * (1 - t) + right_rgba[2] * t)
        a = int(left_rgba[3] * (1 - t) + right_rgba[3] * t)
        gradient.putpixel((x, 0), (r, g, b, a))
    return gradient.resize((width, height))


def draw_gradient_rect(img, x, y, w, h, left_rgba, right_rgba, alpha):
    if w <= 0 or h <= 0:
        return
    left = (left_rgba[0], left_rgba[1], left_rgba[2], alpha)
    right = (right_rgba[0], right_rgba[1], right_rgba[2], alpha)
    rect = make_horizontal_gradient(w, h, left, right)
    img.paste(rect, (x, y), rect)


def load_entries(jsonl_path):
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def pick_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(words, draw, font, max_width):
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def fit_text(text, draw, box_w, box_h):
    words = text.split()
    max_size = max(20, min(96, int(box_h * 0.9)))
    for size in range(max_size, 15, -2):
        font = pick_font(size)
        lines = wrap_text(words, draw, font, box_w)
        max_line_width = 0
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        line_heights = [line_height for _ in lines]
        total_height = line_height * len(lines)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_line_width = max(max_line_width, bbox[2] - bbox[0])
        if max_line_width <= box_w and total_height <= box_h:
            return font, lines, line_heights
    font = pick_font(10)
    lines = wrap_text(words, draw, font, box_w)
    ascent, descent = font.getmetrics()
    line_heights = [ascent + descent for _ in lines]
    return font, lines, line_heights


def draw_caption(entry, output_dir, rng, size_multiple, index):
    width = round_up_multiple(int(entry.get("width", 512)), size_multiple)
    height = round_up_multiple(int(entry.get("height", 512)), size_multiple)
    x = int(entry.get("x", 0))
    y = int(entry.get("y", 0))
    box_w = int(entry.get("box_w", max(16, width // 3)))
    box_h = int(entry.get("box_h", max(16, height // 10)))
    if x + box_w > width:
        box_w = max(16, width - x)
    if y + box_h > height:
        box_h = max(16, height - y)

    align = rng.choice(["left", "center", "right"])
    # Use random solid text color.
    text_color = jitter_color(rng.choice(BASE_PALETTE), rng, jitter=rng.randint(5, 30))
    caption = entry.get("caption", "")

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font, lines, line_heights = fit_text(caption, draw, box_w, box_h)
    total_height = sum(line_heights)
    y_start = y + max(0, (box_h - total_height) // 2)

    # Draw a more visible bounding box background from the palette (black is rare).
    non_black_palette = [c for c in BASE_PALETTE if c != (0, 0, 0)]
    weighted_bg = non_black_palette + non_black_palette
    bg_rgb = jitter_color(rng.choice(weighted_bg), rng, jitter=rng.randint(5, 35))
    bg_alpha = 255
    bg_color = (bg_rgb[0], bg_rgb[1], bg_rgb[2], bg_alpha)
    draw.rectangle([x, y, x + box_w, y + box_h], fill=bg_color)
    stroke_w = 0
    stroke_color = None

    text_layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    y_cursor = max(0, (box_h - total_height) // 2)
    for i, line in enumerate(lines):
        bbox = text_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = line_heights[i]
        if align == "left":
            x_pos = 0
        elif align == "center":
            x_pos = max(0, (box_w - line_width) // 2)
        else:
            x_pos = max(0, box_w - line_width)

        text_draw.text(
            (x_pos, y_cursor),
            line,
            font=font,
            fill=text_color,
            stroke_width=stroke_w,
            stroke_fill=stroke_color,
        )
        y_cursor += line_height

    img.paste(text_layer, (x, y), text_layer)

    out_path = os.path.join(output_dir, f"{index:07d}.png")
    img.save(out_path)
    return out_path, {
        "x": x,
        "y": y,
        "box_w": box_w,
        "box_h": box_h,
        "width": width,
        "height": height,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate caption overlay images.")
    parser.add_argument(
        "--input-jsonl",
        default="/project/llmsvgen/jinmin/SynLayers/caption/captions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-multiple", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_entries(args.input_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    for index, entry in enumerate(entries):
        draw_caption(entry, args.output_dir, rng, args.size_multiple, index)


if __name__ == "__main__":
    main()

