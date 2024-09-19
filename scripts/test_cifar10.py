import torch
from models.simple_cnn_model import SimpleCNN
from utils.data_loader_cifar10 import get_cifar10_loaders

def test_cifar10(batch_size=32):
    _, testloader = get_cifar10_loaders(batch_size=batch_size)

    # 載入已訓練模型
    model = SimpleCNN()
    model.load_state_dict(torch.load('models/cifar10_cnn.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'測試集準確率: {100 * correct / total}%')

if __name__ == "__main__":
    test_cifar10()
