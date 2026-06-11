import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.models import CNNClassifier
from src.utils.data import get_datasets, get_dataset_spec, get_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-class subset expert for System 2")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        required=True,
        help="Global class ids to keep. ORDER MATTERS and must match system2_subset_pipeline --classes.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Lock every RNG so the run is reproducible (model init, dropout, shuffling, split)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_labels(dataset):
    """Read all labels without loading any image. CIFAR exposes .targets, SVHN exposes .labels."""
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "labels"):
        return dataset.labels
    raise ValueError("Dataset exposes neither .targets nor .labels")


class SubsetClassDataset(Dataset):
    """Keep only `indices` and remap their global labels to local 0..K-1 indices.

    The local index of a class is its position in `class_order`, which MUST match the
    order passed to system2_subset_pipeline (it rebuilds local->global the same way).
    """

    def __init__(self, base_dataset, indices, class_order):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.class_to_local = {int(cls): local for local, cls in enumerate(class_order)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, y = self.base_dataset[self.indices[idx]]
        local = self.class_to_local.get(int(y))
        if local is None:
            raise ValueError(f"Label {y} is not in the requested subset {list(self.class_to_local)}")
        return image, local


def filter_indices(dataset, allowed):
    labels = get_labels(dataset)
    allowed_set = set(int(c) for c in allowed)
    return [i for i, y in enumerate(labels) if int(y) in allowed_set]


def build_loaders(dataset_name, root, classes, batch_size, num_workers, val_split, seed):
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")
    if len(set(classes)) != len(classes):
        raise ValueError("--classes must not contain duplicates.")

    # train copy keeps augmentation; val copy uses eval transforms (clean validation metrics)
    train_full_aug = get_train_dataset(dataset_name=dataset_name, root=root, augment=True)
    val_full_eval = get_train_dataset(dataset_name=dataset_name, root=root, augment=False)
    _, test_full = get_datasets(dataset_name=dataset_name, root=root)  # test already uses eval transforms

    subset_idx = filter_indices(train_full_aug, classes)
    if len(subset_idx) == 0:
        raise ValueError(f"No training samples found for classes {classes}.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(subset_idx), generator=generator).tolist()
    val_size = max(1, int(len(subset_idx) * val_split))
    val_pos, train_pos = perm[:val_size], perm[val_size:]
    train_idx = [subset_idx[i] for i in train_pos]
    val_idx = [subset_idx[i] for i in val_pos]
    test_idx = filter_indices(test_full, classes)

    train_dataset = SubsetClassDataset(train_full_aug, train_idx, classes)
    val_dataset = SubsetClassDataset(val_full_eval, val_idx, classes)
    test_dataset = SubsetClassDataset(test_full, test_idx, classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / total, correct / total


def save_history(history, history_json_path, history_csv_path):
    history_json_path.parent.mkdir(parents=True, exist_ok=True)
    with history_json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with history_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def save_curves(history, loss_curve_path, accuracy_curve_path):
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], label="Train Loss")
    plt.plot(epochs, [r["val_loss"] for r in history], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Subset Expert Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_acc"] for r in history], label="Train Accuracy")
    plt.plot(epochs, [r["val_acc"] for r in history], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Subset Expert Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_curve_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_stem = output_path.stem
    history_json_path = output_path.with_name(f"{base_stem}_history.json")
    history_csv_path = output_path.with_name(f"{base_stem}_history.csv")
    loss_curve_path = output_path.with_name(f"{base_stem}_loss_curve.png")
    accuracy_curve_path = output_path.with_name(f"{base_stem}_accuracy_curve.png")
    best_model_path = output_path.with_name(f"{base_stem}_best.pth")
    summary_path = output_path.with_name(f"{base_stem}_summary.json")

    print("Using device:", device)
    print(f"Dataset: {spec.name} | Subset classes (global -> local): "
          f"{ {int(c): i for i, c in enumerate(args.classes)} }")

    train_loader, val_loader, test_loader, train_size, val_size, test_size = build_loaders(
        dataset_name=args.dataset,
        root=args.data_root,
        classes=args.classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"Train samples: {train_size} | Val samples: {val_size} | Test samples: {test_size}")

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=len(args.classes),
        input_channels=spec.input_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_loss": round(float(val_loss), 6),
            "val_acc": round(float(val_acc), 6),
        })

        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = deepcopy(model.state_dict())
            torch.save(best_state_dict, best_model_path)
            status = "best"
        else:
            epochs_without_improvement += 1
            status = f"no_improve={epochs_without_improvement}/{args.patience}"

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | {status}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    torch.save(model.state_dict(), output_path)
    print(f"Last model saved to {output_path}")

    if best_state_dict is None:
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = len(history)
        best_val_loss = history[-1]["val_loss"]
        torch.save(best_state_dict, best_model_path)

    save_history(history, history_json_path, history_csv_path)
    save_curves(history, loss_curve_path, accuracy_curve_path)

    model.load_state_dict(best_state_dict)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    summary = {
        "dataset": spec.name,
        "classes_global": list(args.classes),
        "class_to_local": {int(c): i for i, c in enumerate(args.classes)},
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "final_test_loss": round(float(test_loss), 6),
        "final_test_acc": round(float(test_acc), 6),
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
        "seed": args.seed,
        "artifacts": {
            "last_model": str(output_path),
            "best_model": str(best_model_path),
            "history_json": str(history_json_path),
            "history_csv": str(history_csv_path),
            "loss_curve": str(loss_curve_path),
            "accuracy_curve": str(accuracy_curve_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best model saved to {best_model_path}")
    print(f"Best checkpoint test eval -> Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
