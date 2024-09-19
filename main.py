import argparse
from scripts.train import train_model
from scripts.test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test Transformer model")
    parser.add_argument('--mode', type=str, default='train', help="Mode: 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_model()
    else:
        print("Invalid mode! Use 'train' or 'test'")
