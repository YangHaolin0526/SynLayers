import argparse
import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont

def create_checkerboard(w, h, grid_size=32):
    """创建透明背景的棋盘格 (Checkerboard pattern)"""
    img = Image.new('RGB', (w, h), (200, 200, 200)) # 浅灰背景
    draw = ImageDraw.Draw(img)
    for x in range(0, w, grid_size):
        for y in range(0, h, grid_size):
            if (x // grid_size + y // grid_size) % 2 == 1:
                draw.rectangle([x, y, x + grid_size, y + grid_size], fill=(255, 255, 255))
    return img

def create_layer_composite(sample_dir, metadata):
    """
    生成 'Result' 视图：
    在棋盘格背景上叠加所有图层，并给每个图层的Box画上边框。
    """
    canvas_size = metadata.get('target_size', 512)
    boxes = metadata.get('boxes', [])
    
    # 1. 创建棋盘格底图
    canvas = create_checkerboard(canvas_size, canvas_size).convert('RGBA')
    draw = ImageDraw.Draw(canvas)

    # 2. 依次叠加图层
    # 假设图层命名格式为 layer_00.png, layer_01.png ...
    # 根据你的文件列表，有时是 layer_0.png 有时是 layer_00.png，这里做个兼容
    
    # 尝试按顺序加载图层
    num_layers = len(boxes)
    for i, box in enumerate(boxes):
        # 尝试寻找对应的图层文件
        layer_candidates = [
            os.path.join(sample_dir, f"layer_{i:02d}.png"), # layer_00.png
            os.path.join(sample_dir, f"layer_{i}.png")      # layer_0.png
        ]
        
        layer_path = None
        for p in layer_candidates:
            if os.path.exists(p):
                layer_path = p
                break
        
        if layer_path:
            try:
                # 加载图层
                layer_img = Image.open(layer_path).convert('RGBA')
                
                # 如果图层尺寸不一致，强制缩放 (通常不需要)
                if layer_img.size != canvas.size:
                    layer_img = layer_img.resize(canvas.size, Image.Resampling.BILINEAR)
                
                # 叠加图层 (使用 Alpha 混合)
                canvas = Image.alpha_composite(canvas, layer_img)
                
                # 画黑色边框强调 Box 区域 (模拟 Result 里的黑框效果)
                # 重新获取 draw 对象
                draw = ImageDraw.Draw(canvas)
                x1, y1, x2, y2 = box
                # 边界保护
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(canvas_size, x2), min(canvas_size, y2)
                
                # 画框 (黑色，宽度2)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0, 255), width=2)
                
            except Exception as e:
                print(f"Error loading layer {i}: {e}")

    return canvas.convert('RGB')

def stitch_images(origin_img, bbox_img, result_img):
    """将三张图横向拼接"""
    w, h = origin_img.size
    
    # 创建大画布 (3倍宽度)
    combined = Image.new('RGB', (w * 3, h), (255, 255, 255))
    
    # 粘贴
    combined.paste(origin_img, (0, 0))
    combined.paste(bbox_img, (w, 0))
    combined.paste(result_img, (w * 2, 0))
    
    # (可选) 添加文字标题: Origin, Bbox, Result
    draw = ImageDraw.Draw(combined)
    try:
        # 尝试加载默认字体，如果失败则不画文字或用默认
        font_size = 40
        # linux常见字体路径，如果报错可以注释掉这部分
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = None

    if font:
        # 在每张图上方画标题 (需要画布上方预留空间，或者直接画在图内)
        # 这里为了简单直接画在图的左上角，带个白色背景
        labels = ["Origin", "Bbox", "Result"]
        for i, label in enumerate(labels):
            draw.rectangle([i*w + 10, 10, i*w + 150, 60], fill=(255,255,255, 200))
            draw.text((i*w + 20, 15), label, fill=(0,0,0), font=font)
            
    return combined

def process_sample(sample_dir):
    """处理单个样本文件夹"""
    json_path = os.path.join(sample_dir, 'inference_meta.json')
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # 1. 寻找 Origin 图 (whole_image.png 或 base_image.png)
    origin_candidates = ['whole_image.png', 'base_image.png', 'origin.png']
    origin_path = None
    for name in origin_candidates:
        p = os.path.join(sample_dir, name)
        if os.path.exists(p):
            origin_path = p
            break
            
    # 2. 寻找 Bbox 图 (layout_vis.png - 上一步代码生成的)
    bbox_path = os.path.join(sample_dir, 'layout_vis.png')
    
    if not origin_path or not os.path.exists(bbox_path):
        print(f"Skipping {sample_dir}: Missing origin or layout_vis.png")
        # 如果你还没运行上一步，可以用 create_layer_composite 暂时顶替 bbox_path
        return

    # 加载图片
    origin_img = Image.open(origin_path).convert('RGB')
    bbox_img = Image.open(bbox_path).convert('RGB')

    # 3. 生成 Result 图 (Layer Composite)
    result_img = create_layer_composite(sample_dir, metadata)

    # 4. 拼接
    final_img = stitch_images(origin_img, bbox_img, result_img)
    
    # 5. 保存
    save_path = os.path.join(sample_dir, 'comparison_view.png')
    final_img.save(save_path)
    print(f"Saved comparison: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    search_pattern = os.path.join(args.dataset_dir, 'sample_*')
    sample_dirs = sorted(glob.glob(search_pattern))

    count = 0
    for sample_dir in sample_dirs:
        if args.num_samples and count >= args.num_samples:
            break
        
        if os.path.isdir(sample_dir):
            process_sample(sample_dir)
            count += 1

if __name__ == '__main__':
    main()