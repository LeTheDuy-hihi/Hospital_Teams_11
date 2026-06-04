import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score
import os

from src.data.dataset import MultimodalSkinDataset
from src.models.multimodal_net import MultimodalNet

def train_model():
    # 1. Cấu hình các tham số (Hyperparameters)
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {DEVICE}")

    # 2. Khởi tạo Tokenizer và Transform
    # Vì sử dụng PhoBERT nên ta import tokenizer của PhoBERT
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Train Transform: Có Data Augmentation để giảm overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Val Transform: Không có Augmentation, chỉ Resize và Normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset từ các file đã chia sẵn
    train_csv = 'dataset/train_dataset.csv'
    val_csv = 'dataset/val_dataset.csv'
    img_dir = 'dataset/all_images'
    
    train_dataset = MultimodalSkinDataset(csv_file=train_csv, img_dir=img_dir, tokenizer=tokenizer, transform=train_transform)
    val_dataset = MultimodalSkinDataset(csv_file=val_csv, img_dir=img_dir, tokenizer=tokenizer, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Khởi tạo Mô hình, Hàm Loss và Optimizer
    # Tập HAM10000 có 7 loại bệnh (num_classes=7)
    model = MultimodalNet(num_classes=7).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Lưu trữ lịch sử huấn luyện để vẽ biểu đồ
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    # 5. Vòng lặp Huấn luyện (Training Loop)
    print("\nBắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        for batch in loop:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 6. Đánh giá trên tập Val (Validation)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
            for batch in val_loop:
                images = batch['image'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1} kết quả:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f} | Val Recall: {val_recall:.4f} | Val F1-Score: {val_f1:.4f}")

        # Lưu mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Tự động phát hiện nếu đang chạy trên Colab thì lưu thẳng vào Google Drive
            save_dir = 'models'
            if os.path.exists('/content/drive/MyDrive'):
                save_dir = '/content/drive/MyDrive/DU_AN_BTL_Models'
                
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, 'best_multimodal_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"-> Đã lưu model tốt nhất tại: {save_path}")

    # 7. Vẽ biểu đồ huấn luyện (Visualize)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Biểu đồ Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Biểu đồ Accuracy')
    plt.legend()
    
    plot_path = 'training_history.png'
    if os.path.exists('/content/drive/MyDrive'):
        plot_path = '/content/drive/MyDrive/DU_AN_BTL_Models/training_history.png'
    plt.savefig(plot_path)
    print(f"\nHoàn tất huấn luyện! Đã lưu biểu đồ vào {plot_path}")

if __name__ == '__main__':
    train_model()
