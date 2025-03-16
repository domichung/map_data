import requests
from .env import key

API_KEY = key
# print(API_KEY)

def fetch_satellite_image(lat, lng, width, height, zoom):
    # 向 Google Maps API 發送請求，獲取指定座標的衛星圖片
    response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&format=png&&zoom={zoom}&size={width}x{height}&maptype=satellite&key={key}')
    if response.status_code == 200:
        return response.content
    else:
        print(f"無法抓取圖片 (HTTP 狀態碼 {response.status_code})")
        return None