import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def augment_and_save_images(root_dir):
    for root, dirs, files in os.walk(root_dir):
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for file in image_files:
            file_path = os.path.join(root, file)
            
            try:
                # 開啟原始圖片
                img = Image.open(file_path)
                img = img.convert('RGB')
                
                filename, ext = os.path.splitext(file)

                # 旋轉90度、180度和270度並儲存
                rotation_angles = [45,90,135,180,225,270,315]
                for angle in rotation_angles:
                    # 旋轉圖片
                    rotated_img = img.rotate(angle, expand=True)
                    # 儲存旋轉後的圖片
                    rotated_path = os.path.join(root, f"{filename}_{angle}deg{ext}")
                    rotated_img.save(rotated_path, quality=95)
                
                print(f"已處理並儲存旋轉後的圖片：{filename}")
                
            except Exception as e:
                print(f"處理圖片 {file} 時發生錯誤：{str(e)}")

if __name__ == "__main__":
    # 設定包含訓練資料的根目錄
    train_dir = "dataflip"
    
    # 確保目錄存在
    if os.path.exists(train_dir):
        print(f"開始處理 {train_dir} 目錄中的圖片...")
        augment_and_save_images(train_dir)
        print("處理完成！")
    else:
        print(f"錯誤：找不到 {train_dir} 目錄")