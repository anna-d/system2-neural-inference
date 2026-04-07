import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import CIFAR10CNN
from src.utils.data import get_cifar10_loaders
from src.utils.train_utils import evaluate, train_one_epoch


def run_experiment(hidden_dim, learning_rate, device, epochs=10, batch_size=128):
    train_loader, test_loader, _, _ = get_cifar10_loaders(batch_size=batch_size)

    model = CIFAR10CNN(hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for _ in range(epochs):
        _, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)

    total_time = time.time() - start_time
    return train_acc, test_acc, total_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    hidden_dims = [128, 256, 512]
    learning_rates = [0.0005, 0.001, 0.005]
    results = []

    print("\n=== Running Experiments ===\n")
    for hidden_dim in hidden_dims:
        for learning_rate in learning_rates:
            print(f"Experiment: hidden_dim={hidden_dim}, learning_rate={learning_rate}")
            train_acc, test_acc, elapsed = run_experiment(hidden_dim, learning_rate, device)
            print(f"→ Train Acc: {train_acc * 100:.2f}%")
            print(f"→ Test Acc: {test_acc * 100:.2f}%")
            print(f"→ Time: {elapsed:.2f} sec\n")
            results.append((hidden_dim, learning_rate, train_acc, test_acc, elapsed))

    output_file = Path("experiments_results.txt")
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
