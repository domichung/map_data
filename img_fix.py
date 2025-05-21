import os
import random
import json
from PIL import Image, ImageEnhance, ImageOps

def apply_weather(image, weather_type):
    if weather_type == "sunny":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.5)
    elif weather_type == "cloudy":
        gray = ImageOps.grayscale(image)
        return Image.blend(image, gray.convert("RGB"), alpha=0.5)
    elif weather_type == "rainy":
        overlay = Image.new("RGB", image.size, (100, 100, 100))
        return Image.blend(image, overlay, alpha=0.3)
    elif weather_type == "sunset":
        overlay = Image.new("RGB", image.size, (255, 100, 50))
        return Image.blend(image, overlay, alpha=0.3)
    return image

def main():
    flip_lr = input("1. 左右 (y/n): ").lower() == "y"
    flip_ud = input("2. 上下 (y/n): ").lower() == "y"
    sunny = input("3. 晴天 (y/n): ").lower() == "y"
    cloudy = input("4. 多雲 (y/n): ").lower() == "y"
    rainy = input("5. 下雨 (y/n): ").lower() == "y"
    sunset = input("6. 夕陽 (y/n): ").lower() == "y"

    flip_ops = []
    if flip_lr:
        flip_ops.append("flip_lr")
    if flip_ud:
        flip_ops.append("flip_ud")

    weather_ops = []
    if sunny:
        weather_ops.append("sunny")
    if cloudy:
        weather_ops.append("cloudy")
    if rainy:
        weather_ops.append("rainy")
    if sunset:
        weather_ops.append("sunset")

    src_root = "data"
    dst_root = "new_data"
    log = {}

    for folder in os.listdir(src_root):
        folder_path = os.path.join(src_root, folder)
        if not os.path.isdir(folder_path):
            continue

        dst_folder_path = os.path.join(dst_root, folder)
        os.makedirs(dst_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".png"):
                continue

            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path)
            applied_ops = []

            new_image = image.copy()
            if "flip_lr" in flip_ops and random.choice([True, False]):
                new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
                applied_ops.append("flip_lr")
            if "flip_ud" in flip_ops and random.choice([True, False]):
                new_image = new_image.transpose(Image.FLIP_TOP_BOTTOM)
                applied_ops.append("flip_ud")

            if weather_ops:
                selected_weather = random.choice(weather_ops)
                new_image = apply_weather(new_image, selected_weather)
                applied_ops.append(selected_weather)

            save_path = os.path.join(dst_folder_path, filename)
            new_image.save(save_path)

            rel_path = os.path.join(folder, filename)
            log[rel_path] = applied_ops

    with open(os.path.join(dst_root, "augment_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=4)

    print("所有圖像已增強完畢，結果已儲存於 cnn/new_data/ 中。")

if __name__ == "__main__":
    main()
