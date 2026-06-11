import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier
from src.utils.data import get_datasets, get_dataset_spec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--hidden-dim", type=int, default=256)

    # pair 1
    parser.add_argument("--pair1-weights", type=str, required=True)
    parser.add_argument("--pair1-a", type=int, required=True)
    parser.add_argument("--pair1-b", type=int, required=True)

    # pair 2
    parser.add_argument("--pair2-weights", type=str, required=True)
    parser.add_argument("--pair2-a", type=int, required=True)
    parser.add_argument("--pair2-b", type=int, required=True)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="results/system2_multi_eval.txt")

    return parser.parse_args()


def load_model(weights, num_classes, device, hidden_dim, input_channels):
    model = CNNClassifier(hidden_dim=hidden_dim, num_classes=num_classes, input_channels=input_channels)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_dataset = get_datasets(dataset_name=args.dataset, root="data")
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    spec = get_dataset_spec(args.dataset)
    baseline = load_model(args.baseline_weights, spec.num_classes, device, args.hidden_dim, spec.input_channels)

    pair1 = load_model(args.pair1_weights, 2, device, args.hidden_dim, spec.input_channels)
    pair2 = load_model(args.pair2_weights, 2, device, args.hidden_dim, spec.input_channels)

    baseline_correct = 0
    system2_correct = 0
    total = 0

    trigger_total = 0
    trigger_pair1 = 0
    trigger_pair2 = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = baseline(images)
            probs = F.softmax(logits, dim=1)

            top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
            baseline_preds = probs.argmax(dim=1)
            final_preds = baseline_preds.clone()

            for i in range(images.size(0)):
                y_true = labels[i].item()
                pred = baseline_preds[i].item()

                if pred == y_true:
                    baseline_correct += 1

                top1 = top2_idx[i, 0].item()
                top2 = top2_idx[i, 1].item()
                p1 = top2_probs[i, 0].item()
                p2 = top2_probs[i, 1].item()

                low_margin = abs(p1 - p2) < args.threshold
                pair_set = {top1, top2}

                # ---- Pair 1 ----
                if pair_set == {args.pair1_a, args.pair1_b} and low_margin:
                    trigger_total += 1
                    trigger_pair1 += 1

                    out = pair1(images[i].unsqueeze(0))
                    bpred = out.argmax(dim=1).item()

                    final_preds[i] = args.pair1_a if bpred == 0 else args.pair1_b

                # ---- Pair 2 ----
                elif pair_set == {args.pair2_a, args.pair2_b} and low_margin:
                    trigger_total += 1
                    trigger_pair2 += 1

                    out = pair2(images[i].unsqueeze(0))
                    bpred = out.argmax(dim=1).item()

                    final_preds[i] = args.pair2_a if bpred == 0 else args.pair2_b

            system2_correct += (final_preds == labels).sum().item()
            total += labels.size(0)

    baseline_acc = baseline_correct / total
    system2_acc = system2_correct / total

    lines = [
        "System 2 Multi-Pair Evaluation",
        f"Threshold: {args.threshold}",
        "",
        f"Baseline Accuracy: {baseline_acc:.4f}",
        f"System2 Accuracy: {system2_acc:.4f}",
        f"Improvement: {system2_acc - baseline_acc:.4f}",
        "",
        f"Total Triggers: {trigger_total}",
        f"Pair1 Triggers: {trigger_pair1}",
        f"Pair2 Triggers: {trigger_pair2}",
    ]

    text = "\n".join(lines)
    print("\n" + text)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()