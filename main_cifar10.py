import argparse
from scripts.train_cifar10 import train_cifar10
from scripts.test_cifar10 import test_cifar10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test CIFAR-10 model")
    parser.add_argument('--mode', type=str, default='train', help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == 'train':
        train_cifar10()
    elif args.mode == 'test':
        test_cifar10()
    else:
        print("Invalid mode! Use 'train' or 'test'")
