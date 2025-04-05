import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np

import Dm_model

# -------------------- 資料集類別 --------------------
class LocationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self._load_data()

    def _load_data(self):
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    longitude, latitude = map(float, folder_name.split())
                    for filename in os.listdir(folder_path):
                        if filename.endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(folder_path, filename)
                            self.image_paths.append(image_path)
                            self.labels.append(torch.tensor([longitude, latitude], dtype=torch.float32))
                except ValueError:
                    print(f"跳過無效的資料夾名稱: {folder_name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"無法載入圖片: {image_path}, 錯誤: {e}")
            return None, None

if __name__ == '__main__':
    # -------------------- 設定超參數 --------------------
    DATA_DIR = './data'  # 您的資料集目錄
    BATCH_SIZE = 32
    IMG_SIZE = 224
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    VAL_SPLIT = 0.15
    RANDOM_SEED = 42
    SAVE_PATH = 'pytorch_location_model.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 設定資料轉換 --------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # -------------------- 載入資料集並分割 --------------------
    full_dataset = LocationDataset(DATA_DIR, transform=transform)
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # -------------------- 初始化模型、損失函數和優化器 --------------------
    model = Dm_model.LocationCNN().to(DEVICE)
    #model = SimpleCNN().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------- 訓練迴圈 --------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if images is None:
                continue
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')

        train_loss = train_loss / len(train_dataset)

        # -------------------- 驗證迴圈 --------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_dataset)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # -------------------- 儲存訓練好的模型 --------------------
    torch.save(model.state_dict(), SAVE_PATH)
    print(f'模型已儲存至: {SAVE_PATH}')