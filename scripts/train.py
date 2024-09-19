import torch
import torch.optim as optim
import torch.nn as nn
from models.transformer_model import VisionTransformer
from utils.data_loader import get_data_loaders

def train_model(num_epochs=10, learning_rate=1e-4, batch_size=32, data_dir='data/'):
    # 初始化資料
    train_loader, _ = get_data_loaders(batch_size=batch_size, data_dir=data_dir)

    # 初始化模型、損失函數與優化器
    model = VisionTransformer(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練過程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    torch.save(model.state_dict(), 'models/vit_model.pth')

if __name__ == "__main__":
    train_model()
