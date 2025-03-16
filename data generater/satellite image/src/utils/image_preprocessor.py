import numpy as np

# 剪裁圖片中google logo，並保持正方形
def cut_img(img, len):
    height, width = img.shape[:2]
    left = len
    top = len
    right = width - len
    bottom = height - len 
    return img[top:bottom, left:right]  # 使用 NumPy 切片裁剪
