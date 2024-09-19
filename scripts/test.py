import torch
from models.transformer_model import VisionTransformer
from utils.data_loader import get_data_loaders

def test_model(batch_size=32, data_dir='data/'):
    _, test_loader = get_data_loaders(batch_size=batch_size, data_dir=data_dir)

    # 載入訓練好的模型
    model = VisionTransformer(num_classes=10)
    model.load_state_dict(torch.load('models/vit_model.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

if __name__ == "__main__":
    test_model()
