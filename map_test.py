import requests
import cv2
import numpy as np
import json
import key_save

API_KEY = key_save.key

# 逢甲操場
center_lat = 24.18052
center_lng = 120.64913

# 圖片大小
width = 1000  # Google Maps Static API 的最小尺寸是 640x640
height = 1000

# 圖片縮放級別
zoom = 19  # 調整這個值來改變地圖的縮放級別

# Google Maps Static API URL，指定 maptype 為 satellite
static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={center_lat},{center_lng}&zoom={zoom}&size={width}x{height}&maptype=satellite&key={API_KEY}"

# 發送請求抓取圖片
response = requests.get(static_map_url)
if response.status_code == 200:
    # 將圖片內容轉換為 numpy array
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    # 使用 cv2 解碼圖片
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # 保存圖片
    cv2.imwrite('a.png', image)
    print("圖片已下載並保存為 'a.png'")
else:
    print("無法抓取圖片")

elevation_url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={center_lat},{center_lng}&key={API_KEY}"

# 發送請求獲取高度信息
response = requests.get(elevation_url)
if response.status_code == 200:
    elevation_data = response.json()
    print(elevation_data)  # 打印 API 響應
    if elevation_data['status'] == 'OK':
        elevation = elevation_data['results'][0]['elevation']
        print(f"高度: {elevation} 米")
    else:
        print("無法獲取高度信息")
else:
    print("無法發送請求獲取高度信息")

# 生成 label 檔案
label_data = {
    'latitude': center_lat,
    'longitude': center_lng,
    'elevation': elevation
}

with open('label.json', 'w') as label_file:
    json.dump(label_data, label_file, indent=4)
    print("label 檔案已生成並保存為 'label.json'")
