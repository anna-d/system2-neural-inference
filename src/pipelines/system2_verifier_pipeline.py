import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import CNNClassifier, Verifier
from src.utils.data import get_datasets, get_dataset_spec


def parse_args():
    parser = argparse.ArgumentParser(description="System 2 with a verifier (Option A)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--verifier-weights", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Confidence trigger: run the verifier when max softmax < threshold.")
    parser.add_argument("--verify-threshold", type=float, default=0.50,
                        help="Verifier accepts the prediction when sigmoid(V) >= this value.")
    parser.add_argument("--on-reject", type=str, default="abstain", choices=["abstain", "top2"],
                        help="Action when the verifier rejects: abstain (reject option) or switch to top-2.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/system2_verifier_eval.txt")
    return parser.parse_args()


def load_models(device, baseline_weights, verifier_weights, spec, hidden_dim, embed_dim, head_dim):
    baseline = CNNClassifier(hidden_dim=hidden_dim, num_classes=spec.num_classes, input_channels=spec.input_channels)
    baseline.load_state_dict(torch.load(baseline_weights, map_location=device))
    baseline.to(device).eval()

    verifier = Verifier(
        num_classes=spec.num_classes, hidden_dim=hidden_dim,
        input_channels=spec.input_channels, embed_dim=embed_dim, head_dim=head_dim,
    )
    verifier.load_state_dict(torch.load(verifier_weights, map_location=device))
    verifier.to(device).eval()
    return baseline, verifier


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spec = get_dataset_spec(args.dataset)
    _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    baseline, verifier = load_models(
        device, args.baseline_weights, args.verifier_weights, spec,
        args.hidden_dim, args.embed_dim, args.head_dim,
    )

    total = 0
    baseline_correct = 0

    trigger_count = 0
    # Error-detection confusion (on triggered samples):
    #   reject = verifier says "not valid", confirm = verifier says "valid"
    tp = 0  # baseline wrong  & verifier rejects  (caught error)
    fn = 0  # baseline wrong  & verifier confirms (missed error)
    fp = 0  # baseline right  & verifier rejects  (false alarm)
    tn = 0  # baseline right  & verifier confirms (correct accept)

    # Outcome accounting (depends on --on-reject)
    answered = 0
    answered_correct = 0
    abstained = 0
    system2_correct = 0          # for top2 mode: accuracy over all images
    corrections = 0              # triggered: wrong -> right (top2 mode)
    regressions = 0              # triggered: right -> wrong (top2 mode)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = baseline(images)
            probs = F.softmax(logits, dim=1)
            top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
            confidences = top2_probs[:, 0]
            baseline_preds = top2_idx[:, 0]

            for i in range(images.size(0)):
                y_true = labels[i].item()
                pred = baseline_preds[i].item()
                conf = confidences[i].item()
                baseline_ok = (pred == y_true)
                if baseline_ok:
                    baseline_correct += 1

                final_pred = pred
                is_abstain = False

                if conf < args.threshold:
                    trigger_count += 1
                    v_logit = verifier(images[i].unsqueeze(0), baseline_preds[i].unsqueeze(0))
                    v_prob = torch.sigmoid(v_logit).item()
                    confirm = v_prob >= args.verify_threshold

                    if baseline_ok and confirm:
                        tn += 1
                    elif baseline_ok and not confirm:
                        fp += 1
                    elif (not baseline_ok) and (not confirm):
                        tp += 1
                    else:
                        fn += 1

                    if not confirm:
                        if args.on_reject == "abstain":
                            is_abstain = True
                        else:  # top2
                            final_pred = top2_idx[i, 1].item()
                            if (not baseline_ok) and (final_pred == y_true):
                                corrections += 1
                            elif baseline_ok and (final_pred != y_true):
                                regressions += 1

                if is_abstain:
                    abstained += 1
                else:
                    answered += 1
                    if final_pred == y_true:
                        answered_correct += 1
                        system2_correct += 1
                total += 1

    baseline_acc = baseline_correct / total

    def pct(n, d):
        return f"{(n / d * 100):.2f}%" if d else "n/a"

    triggered_errors = tp + fn          # baseline errors among triggered
    triggered_correct = tn + fp         # baseline correct among triggered
    recall = tp / triggered_errors if triggered_errors else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    false_reject_rate = fp / triggered_correct if triggered_correct else 0.0

    lines = [
        "System 2 Verifier Evaluation (Option A)",
        f"Dataset: {args.dataset}",
        f"Confidence threshold: {args.threshold} | Verify threshold: {args.verify_threshold} | On reject: {args.on_reject}",
        "",
        f"Baseline Accuracy: {baseline_acc:.4f}",
        f"Trigger Count: {trigger_count}",
        "",
        "Verifier error-detection on triggered subset:",
        f"  Baseline errors (TP+FN): {triggered_errors}",
        f"  Caught errors (TP):      {tp}",
        f"  Missed errors (FN):      {fn}",
        f"  False alarms (FP):       {fp}",
        f"  Correct accepts (TN):    {tn}",
        f"  Error-detection recall:    {pct(tp, triggered_errors)}",
        f"  Error-detection precision: {pct(tp, tp + fp)}",
        f"  False-rejection rate:      {pct(fp, triggered_correct)}",
        "",
    ]

    if args.on_reject == "abstain":
        coverage = answered / total if total else 0.0
        selective_acc = answered_correct / answered if answered else 0.0
        lines += [
            "Reject option (abstain):",
            f"  Abstained: {abstained} ({pct(abstained, total)})",
            f"  Coverage (answered): {pct(answered, total)}",
            f"  Selective accuracy (on answered): {selective_acc:.4f}",
        ]
    else:
        system2_acc = system2_correct / total if total else 0.0
        lines += [
            "Top-2 fallback:",
            f"  System2 Accuracy: {system2_acc:.4f}",
            f"  Absolute Improvement: {system2_acc - baseline_acc:.4f}",
            f"  Corrections (wrong->right): {corrections}",
            f"  Regressions (right->wrong): {regressions}",
            f"  Net improvement: {corrections - regressions}",
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
