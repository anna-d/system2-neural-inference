import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_datasets, get_dataset_spec
from src.utils.train_utils import evaluate, train_one_epoch


class BinaryPairDataset(Dataset):
    def __init__(self, base_dataset, class_a: int, class_b: int):
        self.base_dataset = base_dataset
        self.class_a = class_a
        self.class_b = class_b

        self.indices = []
        for i in range(len(base_dataset)):
            _, y = base_dataset[i]
            if y == class_a or y == class_b:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, y = self.base_dataset[self.indices[idx]]

        if y == self.class_a:
            mapped_y = 0
        elif y == self.class_b:
            mapped_y = 1
        else:
            raise ValueError(f"Unexpected label {y} for binary pair ({self.class_a}, {self.class_b})")

        return image, mapped_y


def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary classifier for a confused class pair")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--class-a", type=int, required=True)
    parser.add_argument("--class-b", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_train_val_test_loaders(
    dataset_name: str,
    root: str,
    class_a: int,
    class_b: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
):
    if class_a == class_b:
        raise ValueError("--class-a and --class-b must be different.")
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")

    train_dataset_aug, test_dataset_base = get_datasets(dataset_name=dataset_name, root=root)
    train_dataset_base, _ = get_datasets(dataset_name=dataset_name, root=root)

    pair_train_aug = BinaryPairDataset(train_dataset_aug, class_a, class_b)
    pair_train_base = BinaryPairDataset(train_dataset_base, class_a, class_b)
    pair_test = BinaryPairDataset(test_dataset_base, class_a, class_b)

    total_train = len(pair_train_aug)
    val_size = max(1, int(total_train * val_split))
    train_size = total_train - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no samples for training. Reduce --val-split.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_train, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_subset = Subset(pair_train_aug, train_indices)
    val_subset = Subset(pair_train_base, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        pair_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, len(train_subset), len(val_subset), len(pair_test)


def save_history(history: list[dict], history_json_path: Path, history_csv_path: Path) -> None:
    history_json_path.parent.mkdir(parents=True, exist_ok=True)

    with history_json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with history_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_curves(history: list[dict], loss_curve_path: Path, accuracy_curve_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    train_accs = [row["train_acc"] for row in history]
    val_accs = [row["val_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Pair Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Binary Pair Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_curve_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
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
    print(f"Dataset: {spec.name} | Binary pair: ({args.class_a}, {args.class_b})")

    train_loader, val_loader, test_loader, train_size, val_size, test_size = build_train_val_test_loaders(
        dataset_name=args.dataset,
        root=args.data_root,
        class_a=args.class_a,
        class_b=args.class_b,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    print(f"Train samples: {train_size} | Val samples: {val_size} | Test samples: {test_size}")

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=2,
        input_channels=spec.input_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_loss": round(float(val_loss), 6),
            "val_acc": round(float(val_acc), 6),
        }
        history.append(row)

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
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}."
            )
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
        "class_a": args.class_a,
        "class_b": args.class_b,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "final_test_loss": round(float(test_loss), 6),
        "final_test_acc": round(float(test_acc), 6),
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
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
    print(f"History saved to {history_json_path} and {history_csv_path}")
    print(f"Loss curve saved to {loss_curve_path}")
    print(f"Accuracy curve saved to {accuracy_curve_path}")
    print(
        f"Best checkpoint evaluation on test set -> "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%"
    )
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()