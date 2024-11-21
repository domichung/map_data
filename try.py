import cv2
import numpy as np

# 讀取圖片
image_a = cv2.imread('a.png')
image_b = cv2.imread('b.png')

# 檢查圖片是否成功讀取
if image_a is None:
    print("Error: Could not read 'aiwan_center_satellite_05.png'")
    exit()

if image_b is None:
    print("Error: Could not read 'taiwan_center_satellite_04.png'")
    exit()

# 獲取圖片的形狀 (高度和寬度)
h_a, w_a, _ = image_a.shape
h_b, w_b, _ = image_b.shape

# 提取 image_b 的最上面一行
top_row_b = image_b[0, :, :]

# 初始化標記是否找到匹配
found_match = False

# 從左到右掃描 image_a 的行，尋找匹配的行
for row in range(h_a):
    if np.array_equal(image_a[row, :w_b, :], top_row_b):
        # 找到匹配行，從這行開始將 image_b 複製到 image_a 的下半部分
        copy_height = min(h_b, h_a - row)  # 確認不會超出 image_a 的高度
        
        # 確保只複製到下半部分
        image_a[row:h_a, :w_b, :] = image_b[:(h_a - row), :, :]
        found_match = True
        break  # 找到後即可退出循環

# 檢查是否找到匹配並顯示結果
if found_match:
    print("已成功將 image_b 的像素複製到 image_a 的下半部分")
else:
    print("未找到匹配行，無法複製 image_b 到 image_a")

# 顯示結果或保存圖片
cv2.imwrite('combined_image.png', image_a)
cv2.imshow('Combined Image', image_a)
cv2.waitKey(0)
cv2.destroyAllWindows()