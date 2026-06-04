import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class MultimodalNet(nn.Module):
    def __init__(self, num_classes=7, text_model_name='vinai/phobert-base'):
        super(MultimodalNet, self).__init__()
        
        # 1. Nhánh Vision (Xử lý ảnh)
        # Sử dụng pretrained ResNet18
        self.vision_model = models.resnet18(pretrained=True)
        # Loại bỏ lớp FC cuối cùng của ResNet để lấy vector đặc trưng
        num_ftrs = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Identity() 
        self.vision_out_dim = num_ftrs

        # 2. Nhánh Text (Xử lý văn bản)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_out_dim = self.text_model.config.hidden_size

        # 3. Lớp Kết hợp (Fusion & Phân loại)
        self.classifier = nn.Sequential(
            nn.Linear(self.vision_out_dim + self.text_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Trích xuất đặc trưng ảnh
        img_features = self.vision_model(images) # [batch_size, vision_out_dim]

        # Trích xuất đặc trưng văn bản
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output là vector đặc trưng đại diện cho toàn bộ câu
        text_features = text_outputs.pooler_output # [batch_size, text_out_dim]

        # Kết hợp (Concatenate)
        fused_features = torch.cat((img_features, text_features), dim=1)

        # Phân loại
        output = self.classifier(fused_features)
        return output
