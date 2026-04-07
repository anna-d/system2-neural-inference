import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.models import CNNClassifier
from src.utils.data import (
    SUPPORTED_DATASETS,
    get_class_names,
    get_data_loaders,
    get_dataset_spec,
    unnormalize_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Save example correct and incorrect predictions")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-correct", type=int, default=10)
    parser.add_argument("--max-wrong", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    weights = args.weights or f"artifacts/cnn_{spec.name}.pth"
    print("Using device:", device)
    print(f"Dataset: {spec.name} ({spec.num_classes} classes)")

    _, test_loader, _, test_dataset = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
        num_workers=args.num_workers,
    )
    class_names = get_class_names(test_dataset, dataset_name=args.dataset)

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
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
                image_to_save = unnormalize_tensor(images[i].detach().cpu(), args.dataset).clamp(0.0, 1.0)
                true_name = class_names[labels[i].item()]
                pred_name = class_names[preds[i].item()]

                if correct_saved < args.max_correct and preds[i] == labels[i]:
                    filename = output_dir / f"correct_{correct_saved}_true_{true_name}_pred_{pred_name}.png"
                    save_image(image_to_save, filename)
                    correct_saved += 1

                if wrong_saved < args.max_wrong and preds[i] != labels[i]:
                    filename = output_dir / f"wrong_{wrong_saved}_true_{true_name}_pred_{pred_name}.png"
                    save_image(image_to_save, filename)
                    wrong_saved += 1

                if correct_saved >= args.max_correct and wrong_saved >= args.max_wrong:
                    print(f"Saved examples in '{output_dir}'")
                    return

    print(f"Saved examples in '{output_dir}'")


if __name__ == "__main__":
    main()
