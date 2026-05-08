import os
import json
import random
import argparse
import math
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# ─── Word pools ───────────────────────────────────────────────────────────────

NOUNS = [
    "Sale", "Shop", "Love", "Star", "Moon", "Dream", "Life", "Home", "Food", "Play",
    "Cafe", "Wine", "Beach", "Garden", "Sunset", "Coffee", "Music", "Dance", "Party", "Night",
    "Summer", "Winter", "Spring", "Autumn", "Ocean", "Forest", "Mountain", "River", "Cloud", "Rain",
    "Flower", "Tree", "Bird", "Fish", "Cat", "Dog", "Lion", "Bear", "Wolf", "Fox",
    "Cake", "Pizza", "Bread", "Candy", "Fruit", "Juice", "Cream", "Honey", "Sugar", "Spice",
    "Design", "Style", "Art", "Craft", "Mode", "Trend", "Chic", "Retro", "Modern", "Classic",
    "Hello", "World", "Welcome", "Enjoy", "Happy", "Smile", "Peace", "Hope", "Luck", "Wish",
    "Travel", "Explore", "Wander", "Roam", "Drift", "Journey", "Path", "Road", "Bridge", "Gate",
    "Open", "Free", "New", "Best", "Fresh", "Pure", "Wild", "Bold", "Cool", "Warm",
    "Magic", "Glow", "Shine", "Spark", "Blaze", "Flash", "Bloom", "Wave", "Breeze", "Storm",
]

ADJECTIVES = [
    "Big", "Small", "Hot", "Cold", "Fast", "Slow", "Old", "New", "Dark", "Bright",
    "Sweet", "Sour", "Soft", "Hard", "Deep", "High", "Long", "Short", "Wide", "Thin",
    "Golden", "Silver", "Royal", "Grand", "Fine", "Rare", "Rich", "Pure", "Fresh", "Crisp",
]

VERBS = [
    "Eat", "Run", "Fly", "Swim", "Jump", "Sing", "Cook", "Read", "Draw", "Write",
    "Love", "Dream", "Play", "Dance", "Shine", "Grow", "Bloom", "Rise", "Glow", "Flow",
]

PHRASES_2_3 = [
    "Happy Days", "Good Vibes", "Stay Wild", "Feel Free", "Dream Big",
    "Love More", "Live Well", "Be Kind", "Stay True", "Go Far",
    "Best Ever", "Top Pick", "New Look", "Big Deal", "Hot Sale",
    "Open Now", "Come In", "Try Me", "Buy One", "Save More",
    "Game On", "Well Done", "High Five", "No Fear", "All Good",
    "Time Out", "Night Sky", "Sea Side", "Sun Rise", "Moon Light",
    "Fresh Start", "Last Chance", "First Love", "Sweet Home", "Good Night",
    "Hello World", "Thank You", "Lets Go", "Wake Up", "Stand Up",
    "50% OFF", "BUY NOW", "FREE GIFT", "LIMITED TIME", "MEMBERS ONLY",
    "GRAND OPENING", "COMING SOON", "SOLD OUT", "BACK SOON", "JUST ARRIVED",
]

PHRASES_4_5 = [
    "Live Laugh Love", "Eat Sleep Repeat", "Born To Be Wild",
    "Just Do It Now", "Follow Your Heart", "Chase Your Dreams",
    "Make It Happen", "Never Give Up", "Enjoy The Moment",
    "Life Is Beautiful", "Keep It Simple", "Less Is More",
    "Home Sweet Home", "Best In Town", "One Of A Kind",
    "Fresh From Oven", "Hand Made With Love", "Limited Edition Only",
    "SAVE UP TO 50%", "FREE SHIPPING TODAY", "ORDER NOW SAVE BIG",
    "Welcome To Our Store", "Thank You So Much", "See You Soon",
    "Open All Day Long", "Happy New Year", "Merry Christmas Sale",
]

# ─── Colors ───────────────────────────────────────────────────────────────────

COLORS = [
    (255, 0, 0), (220, 20, 60), (178, 34, 34), (255, 69, 0), (255, 99, 71),
    (0, 0, 255), (0, 0, 139), (30, 144, 255), (65, 105, 225), (70, 130, 180),
    (0, 128, 0), (34, 139, 34), (0, 100, 0), (50, 205, 50), (46, 139, 87),
    (255, 165, 0), (255, 140, 0), (255, 215, 0), (218, 165, 32), (184, 134, 11),
    (128, 0, 128), (148, 0, 211), (186, 85, 211), (153, 50, 204), (138, 43, 226),
    (255, 20, 147), (255, 105, 180), (219, 112, 147), (199, 21, 133), (255, 0, 255),
    (0, 0, 0), (25, 25, 25), (50, 50, 50), (75, 75, 75), (100, 100, 100),
    (255, 255, 255), (245, 245, 245), (220, 220, 220),
    (0, 128, 128), (0, 206, 209), (64, 224, 208),
    (139, 69, 19), (160, 82, 45), (210, 105, 30), (205, 133, 63),
]

# ─── Fonts ────────────────────────────────────────────────────────────────────

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/lato", "/usr/share/fonts/truetype/liberation"]

def get_all_fonts():
    fonts = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.ttf') and not any(x in f.lower() for x in ['math', 'cmmi', 'cmex', 'cmsy', 'msbm', 'msam', 'dsrom', 'wasy']):
                fonts.append(os.path.join(d, f))
    return sorted(fonts)

ALL_FONTS = get_all_fonts()

# ─── Text generation ─────────────────────────────────────────────────────────

def random_case(text):
    """Randomly transform text casing for diversity."""
    r = random.random()
    if r < 0.25:
        return text.upper()
    elif r < 0.45:
        return text.lower()
    elif r < 0.65:
        return text.title()
    else:
        return text

def random_text():
    r = random.random()
    if r < 0.3:
        text = random.choice(PHRASES_2_3)
    elif r < 0.5:
        text = random.choice(PHRASES_4_5)
    else:
        n_words = random.randint(2, 4)
        pool = NOUNS + ADJECTIVES + VERBS
        words = random.sample(pool, min(n_words, len(pool)))
        text = " ".join(words)
    return random_case(text)

# ─── Image generation ────────────────────────────────────────────────────────

def generate_text_image(idx):
    text = random_text()
    font_path = random.choice(ALL_FONTS)
    font_size = random.randint(36, 120)
    color = random.choice(COLORS)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    temp = Image.new("RGBA", (2048, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x = random.randint(10, 40)
    pad_y = random.randint(8, 30)
    canvas_w = tw + pad_x * 2
    canvas_h = th + pad_y * 2

    canvas_w = max(64, min(canvas_w, 1024))
    canvas_h = max(32, min(canvas_h, 512))

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = (canvas_w - tw) // 2 - bbox[0]
    y = (canvas_h - th) // 2 - bbox[1]

    effect = random.random()

    if effect < 0.25:
        # Drop shadow
        shadow_offset = random.randint(2, 5)
        shadow_color = (0, 0, 0, 100)
        draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_color)
        draw.text((x, y), text, font=font, fill=color + (255,))
    elif effect < 0.45:
        # Outline
        outline_w = random.randint(1, 3)
        outline_color = (255, 255, 255, 255) if sum(color) < 400 else (0, 0, 0, 255)
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=color + (255,))
    elif effect < 0.6:
        # Glow
        glow_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_img)
        glow_draw.text((x, y), text, font=font, fill=color + (80,))
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=4))
        img = Image.alpha_composite(img, glow_img)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, font=font, fill=color + (255,))
    else:
        # Plain
        draw.text((x, y), text, font=font, fill=color + (255,))

    output_path = os.path.join(OUTPUT_DIR, f"{idx:07d}.png")
    img.save(output_path)
    return text

# ─── Main ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = ""

def worker(args):
    idx, seed = args
    random.seed(seed)
    text = generate_text_image(idx)
    return idx, text

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = random.Random(args.seed)
    tasks = [(i, rng.randint(0, 2**31)) for i in range(args.num_images)]

    print(f"Generating {args.num_images} text images with {args.num_workers} workers...")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Available fonts: {len(ALL_FONTS)}")

    captions = {}
    with Pool(processes=args.num_workers) as pool:
        for idx, text in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Generating"):
            captions[idx] = text

    # Save captions JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "captions.jsonl")
    with open(jsonl_path, 'w') as f:
        for idx in sorted(captions.keys()):
            f.write(json.dumps({"id": idx, "caption": captions[idx]}) + "\n")

    print(f"Done! {len(captions)} images saved to {OUTPUT_DIR}")
    print(f"Captions saved to {jsonl_path}")


if __name__ == "__main__":
    main()
