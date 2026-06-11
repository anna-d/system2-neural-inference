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

from src.models import Verifier
from src.utils.data import get_datasets, get_dataset_spec, get_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a class-conditional verifier V(x, y) for System 2 (Option A)")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--baseline-weights", type=str, required=True,
                        help="System 1 checkpoint used to initialise the shared backbone.")
    parser.add_argument("--confusion-matrix", type=str, default=None,
                        help="Path to confusion_<dataset>.npy. Defaults to results/confusion_<dataset>.npy.")
    parser.add_argument("--neg-per-pos", type=int, default=1,
                        help="Hard negatives per positive, drawn from the most-confused classes.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Keep the shared backbone frozen (default: fine-tune it).")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_labels(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "labels"):
        return dataset.labels
    raise ValueError("Dataset exposes neither .targets nor .labels")


def build_confusions(matrix, neg_per_pos, num_classes):
    """For each true class y, return the classes it is most confused with (row y, excl. diagonal)."""
    confusions = {}
    for y in range(num_classes):
        row = matrix[y].astype(float).copy()
        row[y] = -1.0  # exclude self
        order = list(np.argsort(row)[::-1])  # descending by confusion count
        negs = [int(c) for c in order if row[c] > 0]
        if len(negs) < max(neg_per_pos, 1):
            for c in order:
                c = int(c)
                if c != y and c not in negs:
                    negs.append(c)
                if len(negs) >= max(neg_per_pos, 1):
                    break
        if not negs:
            negs = [(y + 1) % num_classes]
        confusions[y] = negs[:max(neg_per_pos, 1)]
    return confusions


class VerifierDataset(Dataset):
    """Expands each image into one positive (x, true_label, 1) and
    `neg_per_pos` hard negatives (x, confused_label, 0)."""

    def __init__(self, base_dataset, image_indices, confusions, neg_per_pos):
        self.base = base_dataset
        self.image_indices = list(image_indices)
        self.confusions = confusions
        self.neg_per_pos = neg_per_pos
        self.per_image = 1 + neg_per_pos

    def __len__(self):
        return len(self.image_indices) * self.per_image

    def __getitem__(self, idx):
        ii = idx // self.per_image
        slot = idx % self.per_image
        image, y = self.base[self.image_indices[ii]]
        y = int(y)
        if slot == 0:
            return image, y, 1.0
        negs = self.confusions[y]
        cand = negs[(slot - 1) % len(negs)]
        return image, int(cand), 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, classes, targets in loader:
        images = images.to(device)
        classes = classes.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        logits = model(images, classes)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, classes, targets in loader:
        images = images.to(device)
        classes = classes.to(device)
        targets = targets.float().to(device)

        logits = model(images, classes)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def save_history(history, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def save_curves(history, loss_path, acc_path):
    epochs = [r["epoch"] for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], label="Train Loss")
    plt.plot(epochs, [r["val_loss"] for r in history], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Verifier Training and Validation Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(loss_path, dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_acc"] for r in history], label="Train Acc")
    plt.plot(epochs, [r["val_acc"] for r in history], label="Validation Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Verifier Training and Validation Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(acc_path, dpi=150); plt.close()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    print("Using device:", device)
    print(f"Dataset: {spec.name} | backbone: {'frozen' if args.freeze_backbone else 'fine-tuned'}")

    confusion_path = Path(args.confusion_matrix) if args.confusion_matrix else Path(f"results/confusion_{args.dataset}.npy")
    if not confusion_path.exists():
        raise FileNotFoundError(
            f"Confusion matrix not found at {confusion_path}. Run evaluate_confusion first."
        )
    matrix = np.load(confusion_path)
    confusions = build_confusions(matrix, args.neg_per_pos, spec.num_classes)

    output_path = Path(args.output) if args.output else Path(f"artifacts/verifier_{spec.name}.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    best_path = output_path.with_name(f"{stem}_best.pth")
    history_json = output_path.with_name(f"{stem}_history.json")
    history_csv = output_path.with_name(f"{stem}_history.csv")
    loss_curve = output_path.with_name(f"{stem}_loss_curve.png")
    acc_curve = output_path.with_name(f"{stem}_accuracy_curve.png")
    summary_path = output_path.with_name(f"{stem}_summary.json")

    # Base images: augmented copy for train, eval-transform copy for validation.
    train_base = get_train_dataset(dataset_name=args.dataset, root=args.data_root, augment=True)
    val_base = get_train_dataset(dataset_name=args.dataset, root=args.data_root, augment=False)

    total_images = len(train_base)
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total_images, generator=generator).tolist()
    val_size = max(1, int(total_images * args.val_split))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_dataset = VerifierDataset(train_base, train_idx, confusions, args.neg_per_pos)
    val_dataset = VerifierDataset(val_base, val_idx, confusions, args.neg_per_pos)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)} "
          f"(neg_per_pos={args.neg_per_pos})")

    model = Verifier(
        num_classes=spec.num_classes,
        hidden_dim=args.hidden_dim,
        input_channels=spec.input_channels,
        embed_dim=args.embed_dim,
        head_dim=args.head_dim,
    )
    model.load_backbone(torch.load(args.baseline_weights, map_location=device))
    model.set_backbone_trainable(not args.freeze_backbone)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
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

        if val_loss < (best_val_loss - args.min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            status = "best"
        else:
            no_improve += 1
            status = f"no_improve={no_improve}/{args.patience}"

        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | {status}")

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    torch.save(model.state_dict(), output_path)
    if best_state is None:
        best_state = deepcopy(model.state_dict())
        best_epoch = len(history)
        best_val_loss = history[-1]["val_loss"]
        torch.save(best_state, best_path)

    save_history(history, history_json, history_csv)
    save_curves(history, loss_curve, acc_curve)

    summary = {
        "dataset": spec.name,
        "backbone": "frozen" if args.freeze_backbone else "fine-tuned",
        "neg_per_pos": args.neg_per_pos,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "best_val_acc": history[best_epoch - 1]["val_acc"] if history else None,
        "seed": args.seed,
        "confusion_matrix": str(confusion_path),
        "baseline_weights": args.baseline_weights,
        "artifacts": {
            "last_model": str(output_path),
            "best_model": str(best_path),
            "history_json": str(history_json),
            "loss_curve": str(loss_curve),
            "accuracy_curve": str(acc_curve),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Last model saved to {output_path}")
    print(f"Best model saved to {best_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
