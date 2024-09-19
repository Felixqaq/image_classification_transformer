import torch
import torch.optim as optim
import torch.nn as nn
from models.simple_cnn_model import SimpleCNN
from utils.data_loader_cifar10 import get_cifar10_loaders

def train_cifar10(num_epochs=10, learning_rate=1e-3, batch_size=32):
    # 設置裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入資料
    trainloader, _ = get_cifar10_loaders(batch_size=batch_size)

    # 初始化模型、損失函數和優化器
    model = SimpleCNN().to(device)  # 將模型移動到 GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 將數據移動到 GPU
            optimizer.zero_grad()  # 重置梯度

            # 前向傳播 + 反向傳播 + 優化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader)}')

    # 儲存模型
    torch.save(model.state_dict(), 'models/cifar10_cnn.pth')
    print('訓練完成並儲存模型。')

if __name__ == "__main__":
    train_cifar10()
