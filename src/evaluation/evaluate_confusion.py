import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_class_names, get_datasets, get_dataset_spec


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

    num_classes = get_dataset_spec(args.dataset).num_classes
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

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    np.save(Path(args.output_dir) / f"confusion_{args.dataset}.npy", cm)

    class_names = list(get_class_names(test_dataset, dataset_name=args.dataset))

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