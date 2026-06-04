import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import warnings

class MultimodalSkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Đọc Ảnh an toàn
        img_id = self.data_frame.iloc[idx]['image_id']
        img_name = os.path.join(self.img_dir, img_id + '.jpg')
        
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            warnings.warn(f"Lỗi khi đọc ảnh {img_name}: {e}. Đang dùng ảnh thay thế hoặc padding...")
            # Tạo một ảnh đen giả mạo để tránh crash
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # 2. Đọc Text
        text_description = self.data_frame.iloc[idx]['text_description']
        tokens = self.tokenizer(
            text_description,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        # 3. Đọc Label (Nhãn của bệnh)
        label = self.data_frame.iloc[idx]['label']

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': img_name,
            'text': text_description
        }
