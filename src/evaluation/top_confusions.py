import argparse
import numpy as np
from pathlib import Path

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--output", type=str, default="results/top_confusions.txt")
    args = parser.parse_args()

    cm = np.load(args.matrix)
    cm = cm.copy()

    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:
                pairs.append((i, j, cm[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    # class names
    if args.dataset == "cifar10":
        class_names = CIFAR10_CLASSES
    else:
        class_names = [str(i) for i in range(cm.shape[0])]

    output_lines = []
    output_lines.append(f"Top {args.top_k} confused class pairs:\n")

    for i, j, count in pairs[:args.top_k]:
        line = f"{class_names[i]} ({i}) -> {class_names[j]} ({j}): {count}"
        output_lines.append(line)

    # print to terminal
    print("\n".join(output_lines))

    # save to file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()