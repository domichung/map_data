import requests
from PIL import Image
from io import BytesIO
import numpy as np
import json
import key_save_dir.key_save as key_save
import cal_tool.cal_lib as cal_lib

def cut_img(img):
    width, height = img.size
    left = 1
    top = 0
    right = width - 1
    bottom = height // 10 * 9
    return img.crop((left, top, right, bottom))

API_KEY = key_save.KEY
center_lat = 24.181281
center_lng = 120.648568
width = 640
height = 640
row = 3 
col = 3
zoom = 20

images = []
image_data = []

for x in range(col):
    images.append([])
    for y in range(row):
        new_lat, new_lng = cal_lib.new_center(center_lat, center_lng, width, height-64, zoom, x, y)
        static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={new_lat},{new_lng}&format=png32&scale=2&zoom={zoom}&size={width}x{height}&maptype=satellite&key={API_KEY}"
        response = requests.get(static_map_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            images[x].append(image)
            original_filename = f'./pic/taiwan_center_satellite_{x}_{y}.png'
            cropped_image = cut_img(image)
            cropped_filename = f'./pic/taiwan_center_satellite_cropped_{x}_{y}.png'
            image.save(original_filename)
            cropped_image.save(cropped_filename)
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
cropped_width, cropped_height = cut_img(images[0][0]).size
total_width = cropped_width * col
total_height = cropped_height * row
big_image = Image.new('RGB', (total_width, total_height))

for x in range(col):
    for y in range(row):
        cropped_img = cut_img(images[x][y])
        x_offset = x * cropped_width
        y_offset = y * cropped_height
        big_image.paste(cropped_img, (x_offset, y_offset))

big_image.save('./pic/taiwan_center_satellite_combined.png')
print("合成圖片已保存為 'taiwan_center_satellite_combined.png'")

with open('./pic/label.json', 'w') as label_file:
    json.dump(image_data, label_file, indent=2)
    print("label 檔案已生成並保存為 'label.json'")
