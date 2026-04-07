import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_data_loaders, get_dataset_spec
from src.utils.train_utils import evaluate
from src.utils.tta import predict_with_tta


def evaluate_with_threshold(model, dataloader, device, threshold=0.6):
    model.eval()
    correct = 0
    accepted = 0
    dataset_size = len(dataloader.dataset)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            conf, preds = torch.max(probs, dim=1)

            mask = conf >= threshold
            accepted += mask.sum().item()

            if mask.any():
                correct += preds[mask].eq(labels[mask]).sum().item()

    accuracy = correct / accepted if accepted > 0 else 0.0
    coverage = accepted / dataset_size if dataset_size > 0 else 0.0

    return accuracy, coverage


@torch.no_grad()
def evaluate_tta(model, loader, criterion, device=torch.device("cpu")):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        avg_probs, preds = predict_with_tta(model, images)
        loss = F.nll_loss(torch.log(avg_probs.clamp_min(1e-12)), labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a CNN with and without TTA on CIFAR-10, CIFAR-100, or SVHN"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension used by the trained model",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    weights = args.weights or f"artifacts/cnn_{spec.name}.pth"
    print("Using device:", device)
    print(f"Dataset: {spec.name} ({spec.num_classes} classes)")

    _, test_loader, _, _ = get_data_loaders(
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

    criterion = nn.CrossEntropyLoss()

    start = time.time()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    baseline_time = time.time() - start

    start = time.time()
    tta_loss, tta_acc = evaluate_tta(model, test_loader, criterion, device)
    tta_time = time.time() - start

    print("\n=== Test Results ===")
    print(
        f"Baseline CNN     -> Loss: {test_loss:.4f} | Accuracy: {test_acc * 100:.2f}% | Time: {baseline_time:.2f} sec"
    )
    print(
        f"CNN + TTA        -> Loss: {tta_loss:.4f} | Accuracy: {tta_acc * 100:.2f}% | Time: {tta_time:.2f} sec"
    )
    print(
        f"Accuracy change  -> {((tta_acc - test_acc) * 100):+.2f} percentage points"
    )
    if baseline_time > 0:
        print(f"Time multiplier  -> x{(tta_time / baseline_time):.2f}")
    else:
        print("Time multiplier  -> n/a")

    print("\n=== Confidence Threshold Evaluation ===")
    for th in [0.5, 0.6, 0.7, 0.8]:
        acc, cov = evaluate_with_threshold(model, test_loader, device, threshold=th)
        print(
            f"Threshold {th:.1f} -> Accuracy: {acc * 100:.2f}% | Coverage: {cov * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
