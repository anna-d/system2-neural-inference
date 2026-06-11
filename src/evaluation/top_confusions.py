import argparse
from pathlib import Path

import numpy as np
import torchvision

from src.utils.data import get_class_names as resolve_class_names


def get_class_names(dataset: str, data_root: str):
    """Resolve class names via the central helper in src.utils.data.

    CIFAR exposes real class names through the dataset object, so we load it
    (no transforms needed) and let resolve_class_names read them; SVHN has no
    names and falls back to the dataset spec (digits 0-9).
    """
    if dataset in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        ds = cls(root=data_root, train=False, download=True)
        return list(resolve_class_names(ds, dataset_name=dataset))
    return list(resolve_class_names(None, dataset_name=dataset))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cm = np.load(args.matrix).copy()
    np.fill_diagonal(cm, 0)

    class_names = get_class_names(args.dataset, args.data_root)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:
                pairs.append((i, j, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:args.top_k]

    lines = [f"Top {args.top_k} confused class pairs for {args.dataset}:\n"]
    for i, j, count in top_pairs:
        lines.append(f"{class_names[i]} ({i}) -> {class_names[j]} ({j}): {count}")

    text = "\n".join(lines)
    print(text)

    output_path = args.output
    if output_path is None:
        output_path = f"results/top_confusions_{args.dataset}.txt"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()