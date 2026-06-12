import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import CNNClassifier, Verifier
from src.utils.data import get_datasets, get_dataset_spec
from src.utils.selective import budget_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="System 2: verifier gates a pair expert (Option A + B)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--baseline-weights", type=str, required=True)
    parser.add_argument("--pair-weights", type=str, required=True)
    parser.add_argument("--verifier-weights", type=str, required=True)
    parser.add_argument("--class-a", type=int, required=True)
    parser.add_argument("--class-b", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Confidence trigger: consider System 2 when max softmax < threshold.")
    parser.add_argument("--verify-threshold", type=float, default=0.50,
                        help="Verifier accepts the baseline prediction when sigmoid(V) >= this value.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Calibration temperature for the baseline (confidence trigger).")
    parser.add_argument("--verifier-temperature", type=float, default=1.0,
                        help="Calibration temperature for the verifier (sigmoid).")
    parser.add_argument("--budget", type=float, default=None,
                        help="Coverage budget tau in (0,1]: trigger the most uncertain tau fraction; overrides --threshold.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/system2_verifier_expert_eval.txt")
    return parser.parse_args()


def load_models(device, args, spec):
    baseline = CNNClassifier(hidden_dim=args.hidden_dim, num_classes=spec.num_classes,
                             input_channels=spec.input_channels)
    baseline.load_state_dict(torch.load(args.baseline_weights, map_location=device))
    baseline.to(device).eval()

    pair = CNNClassifier(hidden_dim=args.hidden_dim, num_classes=2, input_channels=spec.input_channels)
    pair.load_state_dict(torch.load(args.pair_weights, map_location=device))
    pair.to(device).eval()

    verifier = Verifier(num_classes=spec.num_classes, hidden_dim=args.hidden_dim,
                        input_channels=spec.input_channels, embed_dim=args.embed_dim, head_dim=args.head_dim)
    verifier.load_state_dict(torch.load(args.verifier_weights, map_location=device))
    verifier.to(device).eval()
    return baseline, pair, verifier


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spec = get_dataset_spec(args.dataset)
    _, test_dataset = get_datasets(dataset_name=args.dataset, root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    baseline, pair, verifier = load_models(device, args, spec)

    total = 0
    baseline_correct = 0
    system2_correct = 0

    trigger_count = 0          # confidence-triggered and baseline pred in pair
    verifier_rejected = 0      # V said "wrong" -> expert ran
    verifier_confirmed = 0     # V said "ok"   -> kept baseline

    # Verifier-gate quality on the triggered subset (did it send the right cases to the expert?)
    gate_tp = 0  # baseline wrong & verifier rejected (correctly sent to expert)
    gate_fp = 0  # baseline right & verifier rejected (sent unnecessarily)
    gate_fn = 0  # baseline wrong & verifier confirmed (missed error)
    gate_tn = 0  # baseline right & verifier confirmed (correctly kept)

    # On the triggered subset (the chain's net effect comes only from expert calls)
    triggered_baseline_correct = 0
    triggered_system2_correct = 0
    corrections = 0            # baseline wrong -> expert right
    regressions = 0            # baseline right -> expert wrong

    if args.budget is not None:
        args.threshold = budget_threshold(baseline, test_loader, device, args.budget, "confidence", args.temperature)
        print(f"Budget {args.budget}: using confidence threshold {args.threshold:.4f}")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            baseline_logits = baseline(images)
            baseline_probs = F.softmax(baseline_logits / args.temperature, dim=1)
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

                    # Verifier gate: only call the expert if the verifier rejects the prediction.
                    v_logit = verifier(images[i].unsqueeze(0), baseline_preds[i].unsqueeze(0))
                    v_prob = torch.sigmoid(v_logit / args.verifier_temperature).item()
                    confirm = v_prob >= args.verify_threshold
                    baseline_ok = (baseline_pred == y_true)

                    if confirm:
                        verifier_confirmed += 1
                        # keep baseline_pred (final_preds already equals it)
                    else:
                        verifier_rejected += 1
                        pair_logits = pair(images[i].unsqueeze(0))
                        pair_pred_local = pair_logits.argmax(dim=1).item()
                        pair_pred_global = args.class_a if pair_pred_local == 0 else args.class_b
                        final_preds[i] = pair_pred_global

                    # Gate quality (reject = sent to expert)
                    if (not baseline_ok) and (not confirm):
                        gate_tp += 1
                    elif baseline_ok and (not confirm):
                        gate_fp += 1
                    elif (not baseline_ok) and confirm:
                        gate_fn += 1
                    else:
                        gate_tn += 1

                    system2_ok = (final_preds[i].item() == y_true)
                    triggered_baseline_correct += int(baseline_ok)
                    triggered_system2_correct += int(system2_ok)
                    if (not baseline_ok) and system2_ok:
                        corrections += 1
                    elif baseline_ok and (not system2_ok):
                        regressions += 1

            system2_correct += (final_preds == labels).sum().item()
            total += labels.size(0)

    baseline_acc = baseline_correct / total
    system2_acc = system2_correct / total

    def pct(n, d):
        return f"{(n / d * 100):.2f}%" if d else "n/a"

    net = corrections - regressions

    lines = [
        "System 2 Verifier + Expert Evaluation (Option A gates B)",
        f"Dataset: {args.dataset}",
        f"Pair: ({args.class_a}, {args.class_b})",
        f"Confidence threshold: {args.threshold} | Verify threshold: {args.verify_threshold}",
        "",
        f"Baseline Accuracy: {baseline_acc:.4f}",
        f"System2 Accuracy: {system2_acc:.4f}",
        f"Absolute Improvement: {system2_acc - baseline_acc:.4f}",
        "",
        f"Triggered (confidence + pred in pair): {trigger_count}",
        f"  Verifier rejected -> expert ran: {verifier_rejected}",
        f"  Verifier confirmed -> kept baseline: {verifier_confirmed}",
        "",
        "Verifier-gate quality on triggered subset (reject = sent to expert):",
        f"  Correctly sent (baseline wrong & rejected, TP): {gate_tp}",
        f"  Sent unnecessarily (baseline right & rejected, FP): {gate_fp}",
        f"  Missed errors (baseline wrong & confirmed, FN): {gate_fn}",
        f"  Correctly kept (baseline right & confirmed, TN): {gate_tn}",
        f"  Error-detection recall:    {pct(gate_tp, gate_tp + gate_fn)}",
        f"  Error-detection precision: {pct(gate_tp, gate_tp + gate_fp)}",
        f"  False-rejection rate:      {pct(gate_fp, gate_fp + gate_tn)}",
        "",
        "On triggered subset:",
        f"  Baseline correct: {triggered_baseline_correct} ({pct(triggered_baseline_correct, trigger_count)})",
        f"  System2 correct:  {triggered_system2_correct} ({pct(triggered_system2_correct, trigger_count)})",
        f"  Corrections (wrong->right): {corrections}",
        f"  Regressions (right->wrong): {regressions}",
        f"  Net improvement: {net}",
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
