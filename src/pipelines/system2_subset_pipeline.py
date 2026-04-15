import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--expert-weights", type=str, required=True)
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--mass-threshold", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/system2_subset_eval.txt")
    return parser.parse_args()


def load_models(device, baseline_weights, expert_weights, num_classes):
    baseline_model = CNNClassifier(num_classes=10, input_channels=3)
    baseline_model.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline_model.to(device)
    baseline_model.eval()

    expert_model = CNNClassifier(num_classes=num_classes, input_channels=3)
    expert_model.load_state_dict(torch.load(expert_weights, map_location=device))
    expert_model.to(device)
    expert_model.eval()

    return baseline_model, expert_model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    subset_classes = list(args.classes)
    subset_set = set(subset_classes)
    local_to_global = {i: cls for i, cls in enumerate(subset_classes)}

    _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    baseline_model, expert_model = load_models(
        device=device,
        baseline_weights=args.baseline_weights,
        expert_weights=args.expert_weights,
        num_classes=len(subset_classes),
    )

    baseline_correct = 0
    system2_correct = 0
    total = 0

    trigger_count = 0
    trigger_on_subset_cases = 0
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
                subset_mass = baseline_probs[i, subset_classes].sum().item()

                if baseline_pred == y_true:
                    baseline_correct += 1

                if confidence < args.threshold and subset_mass > args.mass_threshold:
                    trigger_count += 1

                    if y_true in subset_set:
                        trigger_on_subset_cases += 1

                    expert_logits = expert_model(images[i].unsqueeze(0))
                    expert_pred_local = expert_logits.argmax(dim=1).item()
                    expert_pred_global = local_to_global[expert_pred_local]

                    final_preds[i] = expert_pred_global

                    if expert_pred_global == y_true:
                        trigger_and_correct += 1

            system2_correct += (final_preds == labels).sum().item()
            total += labels.size(0)

    baseline_acc = baseline_correct / total
    system2_acc = system2_correct / total

    lines = [
        "System 2 Subset Expert Evaluation",
        f"Dataset: {args.dataset}",
        f"Subset classes: {subset_classes}",
        f"Threshold: {args.threshold}",
        f"Mass Threshold: {args.mass_threshold}",
        "",
        f"Baseline Accuracy: {baseline_acc:.4f}",
        f"System2 Accuracy: {system2_acc:.4f}",
        f"Absolute Improvement: {system2_acc - baseline_acc:.4f}",
        "",
        f"System2 Trigger Count: {trigger_count}",
        f"Triggers on True Subset Cases: {trigger_on_subset_cases}",
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