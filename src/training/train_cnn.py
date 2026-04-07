import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_data_loaders, get_dataset_spec
from src.utils.train_utils import evaluate, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10, CIFAR-100, or SVHN")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    output = args.output or f"artifacts/cnn_{spec.name}.pth"

    print("Using device:", device)
    print(f"Dataset: {spec.name} ({spec.num_classes} classes)")

    train_loader, test_loader, _, _ = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
        num_workers=args.num_workers,
    )

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
