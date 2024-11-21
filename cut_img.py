import cv2
import numpy as np

def cut_img(img):
    rows = len(img)
    cols = len(img[0]) if rows > 0 else 0  

    new_rows = rows // 10 * 9  
    new_cols = cols - 2  

    if new_rows > 0 and new_cols > 0:
        cpy = [
            [img[i][j] for j in range(1, cols - 1)]  
            for i in range(new_rows)  
        ]
    else:
        cpy = []  

    return cpy

image_a = cv2.imread('a.png')

newimg = np.array(cut_img(image_a), dtype=np.uint8)

cv2.imshow('Combined Image', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()