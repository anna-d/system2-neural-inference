import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "svhn"],
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
    )
    return parser.parse_args()


def get_num_classes(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "svhn":
        return 10
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_class_names(dataset_name: str, test_dataset):
    if dataset_name == "cifar10":
        return list(test_dataset.classes)
    if dataset_name == "cifar100":
        return list(test_dataset.classes)
    if dataset_name == "svhn":
        return [str(i) for i in range(10)]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_labels(dataset_name: str, batch):
    _, labels = batch
    if dataset_name == "svhn":
        labels = labels.long() % 10
    return labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    _, test_dataset = get_datasets(
        dataset_name=args.dataset,
        root=args.data_root,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    num_classes = get_num_classes(args.dataset)
    model = CNNClassifier(num_classes=num_classes, input_channels=3)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.long()
            if args.dataset == "svhn":
                labels = labels % 10

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    np.save(Path(args.output_dir) / f"confusion_{args.dataset}.npy", cm)

    class_names = get_class_names(args.dataset, test_dataset)

    fig_size = (10, 8) if num_classes <= 10 else (20, 20)
    fig, ax = plt.subplots(figsize=fig_size)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False,
        values_format="d" if num_classes <= 10 else None,
    )

    ax.set_title(f"Confusion Matrix - {args.dataset.upper()}")
    plt.tight_layout()
    plt.savefig(Path(args.output_dir) / f"confusion_{args.dataset}.png", dpi=200)
    plt.close(fig)

    accuracy = (all_preds == all_labels).mean()
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved matrix to {Path(args.output_dir) / f'confusion_{args.dataset}.npy'}")
    print(f"Saved plot to {Path(args.output_dir) / f'confusion_{args.dataset}.png'}")


if __name__ == "__main__":
    main()