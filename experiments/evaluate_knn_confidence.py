import argparse
import time
import torch
import torch.nn.functional as F

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_data_loaders, get_dataset_spec


@torch.no_grad()
def build_feature_bank(model, dataloader, device):
    model.eval()

    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        _, feats = model.forward_with_features(images)

        all_features.append(feats)
        all_labels.append(labels)

    feature_bank = torch.cat(all_features, dim=0)
    label_bank = torch.cat(all_labels, dim=0)

    return feature_bank, label_bank


@torch.no_grad()
def knn_predict(query_features, feature_bank, label_bank, k=5, batch_size=256):
    preds_all = []
    num_queries = query_features.size(0)

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        q = query_features[start:end]

        dists = torch.cdist(q, feature_bank)
        _, knn_indices = torch.topk(dists, k=k, largest=False, dim=1)
        knn_labels = label_bank[knn_indices]

        batch_preds = []
        for row in knn_labels:
            values, counts = torch.unique(row, return_counts=True)
            pred = values[counts.argmax()]
            batch_preds.append(pred)

        batch_preds = torch.stack(batch_preds)
        preds_all.append(batch_preds)

    return torch.cat(preds_all, dim=0)


@torch.no_grad()
def evaluate_cnn_and_knn_confidence(
    model,
    test_loader,
    feature_bank,
    label_bank,
    device,
    threshold=0.6,
    k=5,
):
    model.eval()

    total = 0
    cnn_correct = 0

    uncertain_total = 0
    uncertain_cnn_correct = 0
    uncertain_knn_correct = 0
    uncertain_agree = 0

    corrected_cases = 0
    worsened_cases = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, feats = model.forward_with_features(images)
        probs = F.softmax(logits, dim=1)

        conf, cnn_preds = torch.max(probs, dim=1)

        total += labels.size(0)
        cnn_correct += cnn_preds.eq(labels).sum().item()

        uncertain_mask = conf < threshold

        if uncertain_mask.any():
            uncertain_feats = feats[uncertain_mask]
            uncertain_labels = labels[uncertain_mask]
            uncertain_cnn_preds = cnn_preds[uncertain_mask]

            knn_preds = knn_predict(
                uncertain_feats,
                feature_bank,
                label_bank,
                k=k,
            )

            uncertain_total += uncertain_labels.size(0)
            uncertain_cnn_correct += uncertain_cnn_preds.eq(uncertain_labels).sum().item()
            uncertain_knn_correct += knn_preds.eq(uncertain_labels).sum().item()
            uncertain_agree += knn_preds.eq(uncertain_cnn_preds).sum().item()

            corrected_cases += ((uncertain_cnn_preds != uncertain_labels) & (knn_preds == uncertain_labels)).sum().item()
            worsened_cases += ((uncertain_cnn_preds == uncertain_labels) & (knn_preds != uncertain_labels)).sum().item()

    baseline_acc = cnn_correct / total if total > 0 else 0.0
    uncertain_coverage = uncertain_total / total if total > 0 else 0.0
    uncertain_cnn_acc = uncertain_cnn_correct / uncertain_total if uncertain_total > 0 else 0.0
    uncertain_knn_acc = uncertain_knn_correct / uncertain_total if uncertain_total > 0 else 0.0
    agreement = uncertain_agree / uncertain_total if uncertain_total > 0 else 0.0

    return {
        "baseline_acc": baseline_acc,
        "uncertain_coverage": uncertain_coverage,
        "uncertain_cnn_acc": uncertain_cnn_acc,
        "uncertain_knn_acc": uncertain_knn_acc,
        "agreement": agreement,
        "corrected_cases": corrected_cases,
        "worsened_cases": worsened_cases,
        "uncertain_total": uncertain_total,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN + kNN confidence checking on CIFAR-10, CIFAR-100, or SVHN"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension used by the trained model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for uncertain samples")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    weights = args.weights or f"artifacts/cnn_{spec.name}.pth"
    print("Using device:", device)
    print(f"Dataset: {spec.name} ({spec.num_classes} classes)")

    train_loader, test_loader, _, _ = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
        num_workers=args.num_workers,
    )

    model = CNNClassifier(
        hidden_dim=args.hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("\nBuilding feature bank from training set...")
    start = time.time()
    feature_bank, label_bank = build_feature_bank(model, train_loader, device)
    build_time = time.time() - start

    print(f"Feature bank built: {feature_bank.shape[0]} samples, feature_dim={feature_bank.shape[1]}")
    print(f"Build time: {build_time:.2f} sec")

    print("\nEvaluating CNN + kNN confidence check...")
    start = time.time()
    results = evaluate_cnn_and_knn_confidence(
        model=model,
        test_loader=test_loader,
        feature_bank=feature_bank,
        label_bank=label_bank,
        device=device,
        threshold=args.threshold,
        k=args.k,
    )
    eval_time = time.time() - start

    print("\n=== CNN + kNN Confidence Check Results ===")
    print(f"Baseline CNN accuracy                  -> {results['baseline_acc'] * 100:.2f}%")
    print(f"Uncertain sample coverage             -> {results['uncertain_coverage'] * 100:.2f}%")
    print(f"CNN accuracy on uncertain samples     -> {results['uncertain_cnn_acc'] * 100:.2f}%")
    print(f"kNN accuracy on uncertain samples     -> {results['uncertain_knn_acc'] * 100:.2f}%")
    print(f"CNN / kNN agreement                   -> {results['agreement'] * 100:.2f}%")
    print(f"Corrected CNN errors by kNN           -> {results['corrected_cases']}")
    print(f"Worsened correct CNN predictions      -> {results['worsened_cases']}")
    print(f"Number of uncertain samples           -> {results['uncertain_total']} / {results['total']}")
    print(f"Evaluation time                       -> {eval_time:.2f} sec")


if __name__ == "__main__":
    main()
