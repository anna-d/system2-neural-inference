import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--pair-weights", type=str, required=True)
    parser.add_argument("--class-a", type=int, required=True)
    parser.add_argument("--class-b", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/system2_pair_eval.txt")
    return parser.parse_args()


def load_models(device, baseline_weights, pair_weights, dataset):
    num_classes = 100 if dataset == "cifar100" else 10

    baseline_model = CNNClassifier(num_classes=num_classes, input_channels=3)
    baseline_model.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline_model.to(device)
    baseline_model.eval()

    pair_model = CNNClassifier(num_classes=2, input_channels=3)
    pair_model.load_state_dict(torch.load(pair_weights, map_location=device))
    pair_model.to(device)
    pair_model.eval()

    return baseline_model, pair_model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    baseline_model, pair_model = load_models(
        device=device,
        baseline_weights=args.baseline_weights,
        pair_weights=args.pair_weights,
        dataset=args.dataset,
    )

    baseline_correct = 0
    system2_correct = 0
    total = 0

    trigger_count = 0
    trigger_on_true_pair_cases = 0
    trigger_and_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            baseline_logits = baseline_model(images)
            baseline_probs = F.softmax(baseline_logits, dim=1)

            confidences, baseline_preds = baseline_probs.max(dim=1)
            final_preds = baseline_preds.clone()

            for i in range(images.size(0)):
                y_true = labels[i].item()
                baseline_pred = baseline_preds[i].item()
                confidence = confidences[i].item()

                if baseline_pred == y_true:
                    baseline_correct += 1

                if confidence < args.threshold and baseline_pred in (args.class_a, args.class_b):
                    trigger_count += 1

                    if y_true in (args.class_a, args.class_b):
                        trigger_on_true_pair_cases += 1

                    pair_logits = pair_model(images[i].unsqueeze(0))
                    pair_pred_local = pair_logits.argmax(dim=1).item()
                    pair_pred_global = args.class_a if pair_pred_local == 0 else args.class_b

                    final_preds[i] = pair_pred_global

                    if pair_pred_global == y_true:
                        trigger_and_correct += 1

            system2_correct += (final_preds == labels).sum().item()
            total += labels.size(0)

    baseline_acc = baseline_correct / total
    system2_acc = system2_correct / total

    lines = [
        "System 2 Pair Expert Evaluation",
        f"Dataset: {args.dataset}",
        f"Pair: ({args.class_a}, {args.class_b})",
        f"Threshold: {args.threshold}",
        "",
        f"Baseline Accuracy: {baseline_acc:.4f}",
        f"System2 Accuracy: {system2_acc:.4f}",
        f"Absolute Improvement: {system2_acc - baseline_acc:.4f}",
        "",
        f"System2 Trigger Count: {trigger_count}",
        f"Triggers on True Pair Cases: {trigger_on_true_pair_cases}",
        f"Trigger Correct Predictions: {trigger_and_correct}",
    ]

    text = "\n".join(lines)
    print("\n" + text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()