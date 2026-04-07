import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.models import CIFAR10CNN
from src.utils.data import get_cifar10_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Save example correct and incorrect predictions")
    parser.add_argument("--weights", type=str, default="artifacts/cnn_cifar10.pth")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-correct", type=int, default=10)
    parser.add_argument("--max-wrong", type=int, default=10)
    return parser.parse_args()


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalization for image saving."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return tensor * std + mean


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, test_loader, _, test_dataset = get_cifar10_loaders(batch_size=args.batch_size)
    class_names = test_dataset.classes

    model = CIFAR10CNN(hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_saved = 0
    wrong_saved = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if correct_saved < args.max_correct and preds[i] == labels[i]:
                    filename = output_dir / (
                        f"correct_{correct_saved}_true_{class_names[labels[i].item()]}_"
                        f"pred_{class_names[preds[i].item()]}.png"
                    )
                    save_image(unnormalize(images[i].cpu()), filename)
                    correct_saved += 1

                if wrong_saved < args.max_wrong and preds[i] != labels[i]:
                    filename = output_dir / (
                        f"wrong_{wrong_saved}_true_{class_names[labels[i].item()]}_"
                        f"pred_{class_names[preds[i].item()]}.png"
                    )
                    save_image(unnormalize(images[i].cpu()), filename)
                    wrong_saved += 1

                if correct_saved >= args.max_correct and wrong_saved >= args.max_wrong:
                    print(f"Saved examples in '{output_dir}'")
                    return

    print(f"Saved examples in '{output_dir}'")


if __name__ == "__main__":
    main()
