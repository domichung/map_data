import os

# 地理與影像設定
CENTER_LAT = 24.179462   #最左上角圖片緯度參數
CENTER_LNG = 120.648628  #最左上角圖片經度參數
STEP = 0.5               #每次將經緯度往右或往下移動的距離(單位:公尺)
WIDTH = 640              #圖片的寬
HEIGHT = 640             #圖片的高
ROW = 100          
COL = 100                #圖片總數為 ROW*COL

ZOOM = 20                #解析度(他現在很好->別動他)

# 下載圖片的儲存目錄
IMAGES_OUTPUT_DIR = os.path.join('.', 'data')

# 下載圖片的 matadata 儲存目錄 (已修改)
# MATADATA_OUTPUT_DIR = os.path.join('.', 'data', 'metadata')