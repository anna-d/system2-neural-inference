import argparse
import json
import csv
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models import CNNClassifier
from src.utils.data import get_datasets, get_dataset_spec
from src.utils.train_utils import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--class-a", type=int, required=True)
    parser.add_argument("--class-b", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def filter_pair(dataset, a, b):
    indices = [i for i, (_, y) in enumerate(dataset) if y in [a, b]]
    return Subset(dataset, indices)


def split_dataset(dataset, val_split=0.1):
    n = len(dataset)
    val_size = int(n * val_split)
    train_size = n - val_size

    indices = torch.randperm(n).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def plot_curves(history, output_prefix):
    epochs = [h["epoch"] for h in history]

    plt.figure()
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val")
    plt.legend()
    plt.savefig(f"{output_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train")
    plt.plot(epochs, [h["val_acc"] for h in history], label="Val")
    plt.legend()
    plt.savefig(f"{output_prefix}_acc.png")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_datasets(args.dataset)

    train_dataset = filter_pair(train_dataset, args.class_a, args.class_b)
    test_dataset = filter_pair(test_dataset, args.class_a, args.class_b)

    train_dataset, val_dataset = split_dataset(train_dataset, args.val_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = CNNClassifier(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.output.replace(".pth", "_best.pth"))
        else:
            patience_counter += 1

        print(f"[{epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if patience_counter >= args.patience:
            print("Early stopping")
            break

    torch.save(model.state_dict(), args.output)

    plot_curves(history, args.output.replace(".pth", ""))

    with open(args.output.replace(".pth", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()