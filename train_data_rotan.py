import os
import cv2
import numpy as np

input_root = "data"
output_root = "train_rotan_data"
angles = [0, 45, 90, 135, 180, 225, 270, 315]

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            image = cv2.imread(input_path)
            if image is None:
                print(f"讀取失敗: {input_path}")
                continue

            filename, ext = os.path.splitext(file)

            for angle in angles:
                rotated = rotate_image(image, angle)
                new_filename = f"{filename}_{angle}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                cv2.imwrite(output_path, rotated)

print("save as train_rotan_data/")
