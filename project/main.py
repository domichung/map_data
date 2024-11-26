import os
import requests
import cv2
import numpy as np
import json
import key_save_dir.key_save as key_save
import cal_tool.cal_lib as cal_lib

def cut_img(img):
    """裁剪圖片：去除邊框"""
    height, width = img.shape[:2]
    left = 1
    top = 0
    right = width - 1
    bottom = height // 10 * 9
    return img[top:bottom, left:right]  # 使用 NumPy 切片裁剪

API_KEY = key_save.KEY
center_lat = 24.181281
center_lng = 120.648568
width = 640
height = 640
row = 5
col = 5
zoom = 20

output_dir = './pic/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = []
image_data = []

for x in range(col):
    images.append([])
    for y in range(row):
        new_lat, new_lng = cal_lib.new_center(center_lat, center_lng, width-638, height-64, zoom, x, y)
        static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={new_lat},{new_lng}&format=png32&scale=2&zoom={zoom}&size={width}x{height}&maptype=satellite&key={API_KEY}"
        response = requests.get(static_map_url)
        if response.status_code == 200:
            np_img = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            images[x].append(image)
            original_filename = os.path.join(output_dir, f'taiwan_center_satellite_{x}_{y}.png')
            cropped_image = cut_img(image)
            cropped_filename = os.path.join(output_dir, f'taiwan_center_satellite_cropped_{x}_{y}.png')
            cv2.imwrite(original_filename, image)
            cv2.imwrite(cropped_filename, cropped_image)
            print(f"圖片已下載並保存為 '{original_filename}' 和 '{cropped_filename}'")
            image_data.append({
                "latitude": new_lat,
                "longitude": new_lng,
                "original_filename": original_filename,
                "cropped_filename": cropped_filename
            })
        else:
            print(f"無法抓取圖片 (HTTP 狀態碼 {response.status_code}): {static_map_url}")

# 合成圖片
cropped_height, cropped_width = cut_img(images[0][0]).shape[:2]
total_width = cropped_width * col
total_height = cropped_height * row
big_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)  # 黑色背景

for x in range(col):
    for y in range(row):
        cropped_img = cut_img(images[x][y])
        x_offset = x * cropped_width - 648 * x
        y_offset = y * cropped_height
        big_image[y_offset:y_offset+cropped_height, x_offset:x_offset+cropped_width] = cropped_img

combined_filename = os.path.join(output_dir, 'taiwan_center_satellite_combined.png')
cv2.imwrite(combined_filename, big_image)
print(f"合成圖片已保存為 '{combined_filename}'")

label_filename = os.path.join(output_dir, 'label.json')
with open(label_filename, 'w') as label_file:
    json.dump(image_data, label_file, indent=2)
    print(f"label 檔案已生成並保存為 '{label_filename}'")
