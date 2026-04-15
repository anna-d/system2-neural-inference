import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_subset_datasets


class RelabeledSubset(torch.utils.data.Dataset):
    def __init__(self, subset, allowed_classes, dataset_name="cifar10"):
        self.subset = subset
        self.allowed_classes = list(allowed_classes)
        self.dataset_name = dataset_name
        self.class_to_local = {cls: i for i, cls in enumerate(self.allowed_classes)}

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        y = int(y)
        if self.dataset_name == "svhn":
            y = y % 10

        if y not in self.class_to_local:
            raise ValueError(f"Unexpected label {y}")

        return x, self.class_to_local[y]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Subset classes: {args.classes}")

    train_subset, test_subset = get_subset_datasets(
        dataset_name=args.dataset,
        allowed_classes=args.classes,
        root=args.data_root,
    )

    train_dataset = RelabeledSubset(train_subset, args.classes, dataset_name=args.dataset)
    test_dataset = RelabeledSubset(test_subset, args.classes, dataset_name=args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=len(args.classes),
        input_channels=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )

    elapsed = time.time() - start
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()