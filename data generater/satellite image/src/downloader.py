import os
import cv2
import numpy as np
import math
import pymap3d as pm
from api import config, google_api
from utils.image_preprocessor import cut_img
from utils.data_augmentation import data_augmentation

def truncate(f, n):
    # 將數值 f 直接捨去到小數點後 n 位（不進行四捨五入）
    return math.floor(f * 10**n) / 10**n

def meters_to_latlng(metersX, metersY, center_lat, center_lng, center_alt=0):
    # 利用 enu2geodetic 計算：輸入 (e, n, u, center點)
    new_lat, new_lng, _ = pm.enu2geodetic(metersX, metersY, 0, center_lat, center_lng, center_alt)
    
    return truncate(new_lat, 6), truncate(new_lng, 6)

def download_and_save_images():
    # 確保輸出目錄存在
    os.makedirs(config.IMAGES_OUTPUT_DIR, exist_ok=True)

    for x in range(config.COL):
        for y in range(config.ROW):
            # 計算每張影像的經緯度步長
            new_lat, new_lng = meters_to_latlng(config.STEP * x, config.STEP * y, float(config.CENTER_LAT), float(config.CENTER_LNG))
            # print(f"new_lat:{new_lat}, new_lng:{new_lng}")

            try:
                content = google_api.fetch_satellite_image(new_lat, new_lng, config.WIDTH, config.HEIGHT, config.ZOOM)
                if content is None:
                    raise ValueError("API 回傳空內容")

                lat_label = truncate(new_lat, 6)
                lng_label = truncate(new_lng, 6)

                label_dir = os.path.join(config.IMAGES_OUTPUT_DIR, '{:.6f} {:.6f}'.format(truncate(new_lat, 6), truncate(new_lng, 6)))
                os.makedirs(label_dir, exist_ok=True)

                np_img = np.asarray(bytearray(content), np.uint8)
                image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if image is None:
                    raise ValueError("圖片解碼失敗")

                # 剪裁圖片
                image = cut_img(image, 20)

                # 儲存裁剪後的圖片
                image_store_path = os.path.join(label_dir, f'fcu_satellite_{x}_{y}_o.png')
                cv2.imwrite(image_store_path, image)
                # print(f"圖片已保存: {image_store_path}")

                # 資料增強
                # for i, img in enumerate(data_augmentation(image)):
                #     aug_path = os.path.join(label_dir, f'fcu_satellite_{x}_{y}_{i}.png')
                #     cv2.imwrite(aug_path, img)

            except Exception as e:
                print(f"下載失敗 ({new_lat}, {new_lng}): {e}")

