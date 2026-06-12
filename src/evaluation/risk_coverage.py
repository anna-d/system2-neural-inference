import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models import CNNClassifier
from src.utils.data import get_datasets, get_dataset_spec, get_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Risk-coverage curve and AURC for selective prediction")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--score", type=str, default="confidence", choices=["confidence", "margin", "entropy"],
                        help="Uncertainty score used to rank predictions (which to answer first).")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-coverage", type=float, default=None,
                        help="If set, pick the score threshold on validation to hit this coverage, then report test coverage/risk at that fixed threshold (honest protocol).")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_scores(model, loader, device, score, temperature):
    """Return (confidence_like_scores, correct) where HIGHER score = more confident."""
    scores, correct = [], []
    for images, labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images) / temperature, dim=1)
        preds = probs.argmax(dim=1)
        if score == "confidence":
            s = probs.max(dim=1).values
        elif score == "margin":
            top2 = probs.topk(2, dim=1).values
            s = top2[:, 0] - top2[:, 1]
        else:  # entropy -> negative entropy so that higher = more confident
            s = (probs * torch.log(probs + 1e-12)).sum(dim=1)  # = -H
        scores.append(s.cpu().numpy())
        correct.append((preds.cpu() == labels).numpy())
    return np.concatenate(scores), np.concatenate(correct)


def risk_coverage_curve(scores, correct, coverages):
    """Answer the most-confident fraction at each coverage; return list of (coverage, risk, threshold)."""
    order = np.argsort(-scores)            # most confident first
    correct_sorted = correct[order]
    scores_sorted = scores[order]
    n = len(scores)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        answered = correct_sorted[:k]
        risk = 1.0 - answered.mean()
        threshold = float(scores_sorted[k - 1])   # lowest score still answered
        rows.append((c, float(risk), threshold))
    return rows


def aurc(scores, correct):
    """Empirical Area Under the Risk-Coverage curve (lower is better)."""
    order = np.argsort(-scores)
    correct_sorted = correct[order].astype(float)
    n = len(scores)
    cum_err = np.cumsum(1.0 - correct_sorted)
    risks = cum_err / np.arange(1, n + 1)      # risk at each coverage k/n
    return float(risks.mean())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    print(f"Using device: {device}")
    print(f"Risk-coverage on {spec.name} | score: {args.score}")

    model = CNNClassifier(hidden_dim=args.hidden_dim, num_classes=spec.num_classes,
                          input_channels=spec.input_channels)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_scores, test_correct = collect_scores(model, test_loader, device, args.score, args.temperature)

    coverages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = risk_coverage_curve(test_scores, test_correct, coverages)
    test_aurc = aurc(test_scores, test_correct)

    lines = [
        "Risk-Coverage Evaluation",
        f"Dataset: {spec.name} | Score: {args.score}",
        f"Full-coverage accuracy: {test_correct.mean() * 100:.2f}%",
        f"AURC (lower is better): {test_aurc:.4f}",
        "",
        "Coverage | Risk (error on answered) | Threshold",
    ]
    for c, risk, thr in rows:
        lines.append(f"  {c:.2f}     | {risk:.4f}                   | {thr:.4f}")

    # Honest protocol: pick threshold on validation for a target coverage, evaluate on test.
    if args.target_coverage is not None:
        val_base = get_train_dataset(dataset_name=args.dataset, root=args.data_root, augment=False)
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(val_base), generator=generator).tolist()
        val_size = max(1, int(len(val_base) * args.val_split))
        val_loader = DataLoader(Subset(val_base, perm[:val_size]), batch_size=args.batch_size,
                                shuffle=False, num_workers=2)
        val_scores, _ = collect_scores(model, val_loader, device, args.score, args.temperature)
        # threshold = the (1 - target_coverage) quantile of validation scores (answer the top target_coverage)
        thr = float(np.quantile(val_scores, 1.0 - args.target_coverage))
        answered_mask = test_scores >= thr
        cov = float(answered_mask.mean())
        risk = float(1.0 - test_correct[answered_mask].mean()) if answered_mask.any() else 0.0
        lines += [
            "",
            f"Validation-selected threshold for target coverage {args.target_coverage}:",
            f"  Threshold (from validation): {thr:.4f}",
            f"  Test coverage at this threshold: {cov:.4f}",
            f"  Test risk on answered: {risk:.4f}",
        ]

    text = "\n".join(lines)
    print("\n" + text)

    output_path = Path(args.output) if args.output else Path(f"results/risk_coverage_{spec.name}_{args.score}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text + "\n")
    # CSV for plotting
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coverage", "risk", "threshold"])
        writer.writerows(rows)
    print(f"\nSaved to {output_path} and {csv_path}")


if __name__ == "__main__":
    main()
