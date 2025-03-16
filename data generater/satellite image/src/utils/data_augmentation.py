import cv2
import numpy as np

# 以下有需要再補
# Resize/Rescale
# Cropping
# Padding
# Random Affine

def data_augmentation (image):
    augmentation_images = []

    # 旋轉 90 180 270 
    augmentation_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmentation_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmentation_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 水平、垂直翻轉
    augmentation_images.append(cv2.flip(image, 1))
    augmentation_images.append(cv2.flip(image, 0))
    augmentation_images.append(cv2.flip(image, -1))

    # Gaussian Blur
    augmentation_images.append(cv2.GaussianBlur(image, (3,3), 0, 0))

    result_images = augmentation_images.copy()

    for img in augmentation_images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Brightness
        # 亮度增強
        bright_hsv = hsv.copy()
        bright_hsv[:, :, 2] = np.clip(bright_hsv[:, :, 2] * 1.5, 0, 255).astype(np.uint8)
        bright_img = cv2.cvtColor(bright_hsv, cv2.COLOR_HSV2BGR)
        
        # 亮度降低
        dark_hsv = hsv.copy()
        dark_hsv[:, :, 2] = np.clip(dark_hsv[:, :, 2] * 0.5, 0, 255).astype(np.uint8)
        dark_img = cv2.cvtColor(dark_hsv, cv2.COLOR_HSV2BGR)
        
        # Saturation
        # 增強飽和度
        high_saturation_hsv = hsv.copy()
        high_saturation_hsv[:, :, 1] = np.clip(high_saturation_hsv[:, :, 1] * 1.5, 0, 225)
        high_saturation_img = cv2.cvtColor(high_saturation_hsv, cv2.COLOR_HSV2BGR)

        # 降低飽和度
        low_saturation_hsv = hsv.copy()
        low_saturation_hsv[:, :, 1] = np.clip(low_saturation_hsv[:, :, 1] * 0.5, 0, 225)
        low_saturation_img = cv2.cvtColor(low_saturation_hsv, cv2.COLOR_HSV2BGR)

        result_images.extend([bright_img, dark_img, high_saturation_img, low_saturation_img])

        result_images.append(simulate_sunny(img))
        result_images.append(simulate_cloudy(img))
        result_images.append(simulate_rainy(img))
        result_images.append(simulate_sunset(img))

    return result_images

# 模擬天氣
def simulate_sunny(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255).astype(np.uint8)  # 飽和度增加
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255).astype(np.uint8)  # 增加亮度
    sunny = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return sunny

def simulate_cloudy(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255).astype(np.uint8)  # 降低亮度
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255).astype(np.uint8)  # 降低飽和度
    cloudy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cloudy

def simulate_rainy(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255).astype(np.uint8)  # 降低亮度
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255).astype(np.uint8)  # 降低飽和度
    rainy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    b, g, r = cv2.split(rainy)
    b = np.clip(b * 1.1, 0, 255).astype(np.uint8)
    rainy = cv2.merge((b, g, r))
    return rainy

def simulate_sunset(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int32) + 10) % 180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255).astype(np.uint8)
    sunset = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sunset = cv2.convertScaleAbs(sunset, alpha=1.1, beta=15)
    return sunset