import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random

import Dm_model

# -------------------- 設定資料轉換 --------------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- 載入所有資料的路徑和標籤 --------------------
def load_all_data_paths(data_dir):
    image_paths = []
    labels = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                longitude, latitude = map(float, folder_name.split())
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, filename)
                        image_paths.append(image_path)
                        labels.append(torch.tensor([longitude, latitude], dtype=torch.float32))
            except ValueError:
                print(f"跳過無效的資料夾名稱: {folder_name}")
    return image_paths, labels

def preprocess_single_image(image_path, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        print(f"無法載入或預處理圖片: {image_path}, 錯誤: {e}")
        return None

if __name__ == '__main__':
    DATA_DIR = './data'  # 相對於 cnn.py 的路徑
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = 'pro_pytorch_location_model.pth'  # 假設您已經訓練好並儲存了模型

    # 載入所有圖片路徑和標籤
    all_image_paths, all_labels = load_all_data_paths(DATA_DIR)

    if not all_image_paths:
        print("找不到任何圖片資料。請檢查資料集路徑。")
        exit()

    # 隨機選擇一個圖片的索引
    random_index = random.randint(0, len(all_image_paths) - 1)
    random_image_path = all_image_paths[random_index]
    real_label = all_labels[random_index].to(DEVICE).unsqueeze(0) # 保持與模型輸出相同的維度

    # 預處理選中的圖片
    input_tensor = preprocess_single_image(random_image_path, transform, DEVICE)

    if input_tensor is not None:
        # 載入已訓練的模型
        loaded_model = Dm_model.LocationCNN().to(DEVICE)
        try:
            loaded_model.load_state_dict(torch.load(SAVE_PATH))
            loaded_model.eval()
            with torch.no_grad():
                prediction = loaded_model(input_tensor)

            predicted_location = prediction.cpu().numpy()[0]
            real_location = real_label.cpu().numpy()[0]

            print(f"隨機選取的圖片: {os.path.basename(random_image_path)}")
            print(f"真實經度: {real_location[0]:.6f}, 真實緯度: {real_location[1]:.6f}")
            print(f"預測經度: {predicted_location[0]:.6f}, 預測緯度: {predicted_location[1]:.6f}")

        except FileNotFoundError:
            print(f"找不到已訓練的模型: {SAVE_PATH}")
        except Exception as e:
            print(f"載入或預測模型時發生錯誤: {e}")