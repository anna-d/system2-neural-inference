import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_data_loaders, get_dataset_spec
from src.utils.train_utils import evaluate, train_one_epoch


def run_experiment(dataset, hidden_dim, learning_rate, device, epochs=10, batch_size=128, data_root="data"):
    spec = get_dataset_spec(dataset)
    train_loader, test_loader, _, _ = get_data_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
        root=data_root,
    )

    model = CNNClassifier(
        hidden_dim=hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    total_time = time.time() - start_time
    return train_acc, test_acc, total_time


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments on CIFAR-10, CIFAR-100, or SVHN")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Dataset: {args.dataset}")

    hidden_dims = [128, 256, 512]
    learning_rates = [0.0005, 0.001, 0.005]
    results = []

    print("\n=== Running Experiments ===\n")
    for hidden_dim in hidden_dims:
        for learning_rate in learning_rates:
            print(f"Experiment: hidden_dim={hidden_dim}, learning_rate={learning_rate}")
            train_acc, test_acc, elapsed = run_experiment(
                args.dataset,
                hidden_dim,
                learning_rate,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_root=args.data_root,
            )
            print(f"→ Train Acc: {train_acc * 100:.2f}%")
            print(f"→ Test Acc: {test_acc * 100:.2f}%")
            print(f"→ Time: {elapsed:.2f} sec\n")
            results.append((hidden_dim, learning_rate, train_acc, test_acc, elapsed))

    output_file = Path(f"experiments_results_{args.dataset}.txt")
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("hidden_dim | learning_rate | train_acc | test_acc | time_sec\n")
        handle.write("-------------------------------------------------------------\n")
        for hidden_dim, learning_rate, train_acc, test_acc, elapsed in results:
            handle.write(
                f"{hidden_dim:<10} | {learning_rate:<13} | "
                f"{train_acc * 100:>8.2f}% | {test_acc * 100:>8.2f}% | {elapsed:.2f}\n"
            )

    print("All experiments completed.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
