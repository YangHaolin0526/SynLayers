import argparse
import json
import os
import glob
import random
from PIL import Image, ImageDraw

# 定义高对比度配色方案 (R, G, B)
COLORS = [
    (100, 149, 237), # CornflowerBlue
    (255, 105, 180), # HotPink
    (50, 205, 50),   # LimeGreen
    (255, 165, 0),   # Orange
    (147, 112, 219), # MediumPurple
    (0, 206, 209),   # DarkTurquoise
    (255, 99, 71),   # Tomato
    (255, 215, 0),   # Gold
    (138, 43, 226),  # BlueViolet
    (0, 128, 128),   # Teal
    (165, 42, 42),   # Brown
    (70, 130, 180),  # SteelBlue
]

def process_single_sample(json_path):
    """
    读取 json_path，生成 layout_vis.png 并保存在同一目录下
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read {json_path}: {e}")
        return False

    # 1. 获取基础信息
    canvas_size = data.get('target_size', 512)
    boxes = data.get('boxes', [])
    
    # 2. 创建白色背景画布 (RGBA模式以便做透明混合)
    base_img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 255))

    # 3. 遍历并绘制 Box
    for i, box in enumerate(boxes):
        if len(box) < 4: continue # 跳过异常数据
        
        x1, y1, x2, y2 = box
        
        # 简单边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(canvas_size, x2), min(canvas_size, y2)
        
        # 选取颜色
        color_rgb = COLORS[i % len(COLORS)]
        
        # --- A. 绘制半透明填充 (Fill) ---
        # 创建临时透明层
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        # Alpha=50 (约20%不透明度)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=color_rgb + (50,))
        # 混合到底图
        base_img = Image.alpha_composite(base_img, overlay)
        
        # --- B. 绘制实心边框 (Stroke) ---
        draw_outline = ImageDraw.Draw(base_img)
        # Alpha=255 (完全不透明)
        draw_outline.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=3)
        
        # (可选) 可以在这里绘制文字标签
        # draw_outline.text((x1+2, y1+2), str(i), fill=color_rgb + (255,))

    # 4. 保存到同一目录
    save_dir = os.path.dirname(json_path)
    save_name = "bbox.png" # 固定的输出文件名
    save_path = os.path.join(save_dir, save_name)
    
    # 转回 RGB 格式保存
    base_img.convert('RGB').save(save_path)
    return save_path

def main():
    parser = argparse.ArgumentParser(description='Visualize boxes inside each sample folder')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory containing sample_ folders')
    parser.add_argument('--num_samples', type=int, default=None, help='Limit number of samples (optional, default: all)')
    
    args = parser.parse_args()

    # 查找所有 sample_*/inference_meta.json
    # 使用 recursive=True 并不是必须的，因为结构通常只有一层，但为了稳健这里用简单的通配符
    search_pattern = os.path.join(args.dataset_dir, 'sample_*', 'inference_meta.json')
    json_files = sorted(glob.glob(search_pattern))

    if not json_files:
        print(f"No inference_meta.json found in {args.dataset_dir}")
        return

    print(f"Found {len(json_files)} samples.")
    
    # 如果指定了数量限制
    if args.num_samples is not None:
        json_files = json_files[:args.num_samples]
        print(f"Processing first {args.num_samples} samples...")

    success_count = 0
    for json_file in json_files:
        output_path = process_single_sample(json_file)
        if output_path:
            # 打印相对简洁的日志
            print(f"Generated: {output_path}")
            success_count += 1

    print(f"\nCompleted! {success_count} images saved inside their respective folders.")

if __name__ == '__main__':
    main()