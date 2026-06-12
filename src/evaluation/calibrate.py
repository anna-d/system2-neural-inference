import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models import CNNClassifier, Verifier
from src.utils.calibration import ece_binary, ece_multiclass, fit_temperature
from src.utils.data import get_datasets, get_dataset_spec, get_train_dataset
from src.training.train_verifier import VerifierDataset, build_confusions


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature-scale a model and report ECE before/after")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-type", type=str, default="classifier", choices=["classifier", "verifier"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    # verifier-only
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--confusion-matrix", type=str, default=None)
    parser.add_argument("--neg-per-pos", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def val_indices(n, val_split, seed):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    val_size = max(1, int(n * val_split))
    return perm[:val_size]


@torch.no_grad()
def collect_classifier_logits(model, loader, device):
    logits_all, labels_all = [], []
    for images, labels in loader:
        out = model(images.to(device))
        logits_all.append(out.cpu())
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


@torch.no_grad()
def collect_verifier_logits(model, loader, device):
    logits_all, labels_all = [], []
    for images, classes, targets in loader:
        out = model(images.to(device), classes.to(device))
        logits_all.append(out.cpu())
        labels_all.append(targets)
    return torch.cat(logits_all), torch.cat(labels_all)


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    print(f"Using device: {device}")
    print(f"Calibrating {args.model_type} on {spec.name}")

    if args.model_type == "classifier":
        val_base = get_train_dataset(dataset_name=args.dataset, root=args.data_root, augment=False)
        v_idx = val_indices(len(val_base), args.val_split, args.seed)
        _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
        val_loader = DataLoader(Subset(val_base, v_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = CNNClassifier(hidden_dim=args.hidden_dim, num_classes=spec.num_classes,
                              input_channels=spec.input_channels)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device).eval()

        val_logits, val_labels = collect_classifier_logits(model, val_loader, device)
        test_logits, test_labels = collect_classifier_logits(model, test_loader, device)

        temperature = fit_temperature(val_logits, val_labels, mode="multiclass")

        probs_before = F.softmax(test_logits, dim=1).numpy()
        probs_after = F.softmax(test_logits / temperature, dim=1).numpy()
        labels_np = test_labels.numpy()
        ece_before = ece_multiclass(probs_before, labels_np, args.n_bins)
        ece_after = ece_multiclass(probs_after, labels_np, args.n_bins)
        accuracy = float((probs_before.argmax(axis=1) == labels_np).mean())

    else:  # verifier
        confusion_path = Path(args.confusion_matrix) if args.confusion_matrix else Path(f"results/confusion_{args.dataset}.npy")
        if not confusion_path.exists():
            raise FileNotFoundError(f"Confusion matrix not found at {confusion_path}.")
        confusions = build_confusions(np.load(confusion_path), args.neg_per_pos, spec.num_classes)

        val_base = get_train_dataset(dataset_name=args.dataset, root=args.data_root, augment=False)
        v_idx = val_indices(len(val_base), args.val_split, args.seed)
        _, test_base = get_datasets(dataset_name=args.dataset, root=args.data_root)

        val_ds = VerifierDataset(val_base, v_idx, confusions, args.neg_per_pos)
        test_ds = VerifierDataset(test_base, list(range(len(test_base))), confusions, args.neg_per_pos)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = Verifier(num_classes=spec.num_classes, hidden_dim=args.hidden_dim,
                         input_channels=spec.input_channels, embed_dim=args.embed_dim, head_dim=args.head_dim)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.to(device).eval()

        val_logits, val_labels = collect_verifier_logits(model, val_loader, device)
        test_logits, test_labels = collect_verifier_logits(model, test_loader, device)

        temperature = fit_temperature(val_logits, val_labels, mode="binary")

        probs_before = torch.sigmoid(test_logits).numpy()
        probs_after = torch.sigmoid(test_logits / temperature).numpy()
        labels_np = test_labels.numpy().astype(int)
        ece_before = ece_binary(probs_before, labels_np, args.n_bins)
        ece_after = ece_binary(probs_after, labels_np, args.n_bins)
        accuracy = float(((probs_before >= 0.5).astype(int) == labels_np).mean())

    print(f"\nTemperature (fit on validation): {temperature:.4f}")
    print(f"Accuracy (unchanged by scaling): {accuracy * 100:.2f}%")
    print(f"ECE before: {ece_before:.4f}")
    print(f"ECE after:  {ece_after:.4f}")
    print(f"ECE reduction: {ece_before - ece_after:.4f}")

    output_path = Path(args.output) if args.output else Path(args.weights).with_name(
        Path(args.weights).stem + "_temperature.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": spec.name,
            "model_type": args.model_type,
            "weights": args.weights,
            "temperature": round(temperature, 6),
            "accuracy": round(accuracy, 6),
            "ece_before": round(ece_before, 6),
            "ece_after": round(ece_after, 6),
            "n_bins": args.n_bins,
        }, f, indent=2)
    print(f"Saved temperature to {output_path}")


if __name__ == "__main__":
    main()
