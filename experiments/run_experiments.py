import argparse
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.models import CNNClassifier
from src.utils.data import SUPPORTED_DATASETS, get_datasets, get_dataset_spec, get_train_dataset
from src.utils.train_utils import evaluate, train_one_epoch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(dataset, batch_size, data_root, val_split, seed, num_workers=0):
    """Build train/val/test loaders with a single, seeded train/val split.

    The same split is reused for every hyperparameter configuration so the
    comparison between configs is fair. The validation copy uses evaluation
    transforms; the test set is built here but only touched once, at the very end.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")

    train_aug = get_train_dataset(dataset_name=dataset, root=data_root, augment=True)
    val_eval = get_train_dataset(dataset_name=dataset, root=data_root, augment=False)
    _, test_dataset = get_datasets(dataset_name=dataset, root=data_root)

    total = len(train_aug)
    val_size = max(1, int(total * val_split))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_loader = DataLoader(
        Subset(train_aug, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(val_eval, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def run_experiment(train_loader, val_loader, dataset, hidden_dim, learning_rate, device, epochs):
    """Train one configuration and report VALIDATION accuracy (never test)."""
    spec = get_dataset_spec(dataset)
    model = CNNClassifier(
        hidden_dim=hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    train_acc = 0.0
    val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    total_time = time.time() - start_time
    return train_acc, val_acc, total_time, model.state_dict()


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments on CIFAR-10, CIFAR-100, or SVHN")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    print("Using device:", device)
    print(f"Dataset: {args.dataset}")

    train_loader, val_loader, test_loader = build_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        data_root=args.data_root,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    hidden_dims = [128, 256, 512]
    learning_rates = [0.0005, 0.001, 0.005]
    results = []

    best_val_acc = -1.0
    best_config = None
    best_state_dict = None

    print("\n=== Running Experiments (model selection on validation) ===\n")
    for hidden_dim in hidden_dims:
        for learning_rate in learning_rates:
            print(f"Experiment: hidden_dim={hidden_dim}, learning_rate={learning_rate}")
            train_acc, val_acc, elapsed, state_dict = run_experiment(
                train_loader,
                val_loader,
                args.dataset,
                hidden_dim,
                learning_rate,
                device,
                epochs=args.epochs,
            )
            print(f"→ Train Acc: {train_acc * 100:.2f}%")
            print(f"→ Val Acc:   {val_acc * 100:.2f}%")
            print(f"→ Time: {elapsed:.2f} sec\n")
            results.append((hidden_dim, learning_rate, train_acc, val_acc, elapsed))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (hidden_dim, learning_rate)
                best_state_dict = deepcopy(state_dict)

    # Touch the test set exactly once, for the configuration selected on validation.
    criterion = nn.CrossEntropyLoss()
    best_hidden_dim, best_lr = best_config
    best_model = CNNClassifier(
        hidden_dim=best_hidden_dim,
        num_classes=spec.num_classes,
        input_channels=spec.input_channels,
    ).to(device)
    best_model.load_state_dict(best_state_dict)
    _, final_test_acc = evaluate(best_model, test_loader, criterion, device)

    output_file = Path(f"experiments_results_{args.dataset}.txt")
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("hidden_dim | learning_rate | train_acc | val_acc | time_sec\n")
        handle.write("-----------------------------------------------------------\n")
        for hidden_dim, learning_rate, train_acc, val_acc, elapsed in results:
            handle.write(
                f"{hidden_dim:<10} | {learning_rate:<13} | "
                f"{train_acc * 100:>8.2f}% | {val_acc * 100:>7.2f}% | {elapsed:.2f}\n"
            )
        handle.write("\n")
        handle.write(
            f"Best config (selected on validation): hidden_dim={best_hidden_dim}, "
            f"learning_rate={best_lr} | val_acc={best_val_acc * 100:.2f}%\n"
        )
        handle.write(f"Final TEST accuracy of best config: {final_test_acc * 100:.2f}%\n")

    print("All experiments completed.")
    print(
        f"Best config (val): hidden_dim={best_hidden_dim}, learning_rate={best_lr} "
        f"| val_acc={best_val_acc * 100:.2f}%"
    )
    print(f"Final TEST accuracy of best config: {final_test_acc * 100:.2f}%")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
