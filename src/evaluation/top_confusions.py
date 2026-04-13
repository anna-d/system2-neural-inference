import argparse
from pathlib import Path

import numpy as np
import torchvision


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_class_names(dataset: str, data_root: str):
    if dataset == "cifar10":
        return CIFAR10_CLASSES

    if dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        return list(ds.classes)

    if dataset == "svhn":
        return [str(i) for i in range(10)]

    return [str(i) for i in range(100)]


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