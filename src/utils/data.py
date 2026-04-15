from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_channels: int
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    class_names: tuple[str, ...]


_DATASET_SPECS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        input_channels=3,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        class_names=(
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        num_classes=100,
        input_channels=3,
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        class_names=tuple(str(i) for i in range(100)),
    ),
    "svhn": DatasetSpec(
        name="svhn",
        num_classes=10,
        input_channels=3,
        image_size=32,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        class_names=tuple(str(i) for i in range(10)),
    ),
}


SUPPORTED_DATASETS: tuple[str, ...] = tuple(_DATASET_SPECS.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in _DATASET_SPECS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}")
    return _DATASET_SPECS[key]


def build_transforms(dataset_name: str, train: bool) -> transforms.Compose:
    spec = get_dataset_spec(dataset_name)
    transform_steps: list[transforms.Compose | transforms.Normalize | transforms.RandomHorizontalFlip | transforms.RandomCrop | transforms.ToTensor] = []

    if train:
        if dataset_name.lower() in {"cifar10", "cifar100"}:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(spec.image_size, padding=4),
                ]
            )
        elif dataset_name.lower() == "svhn":
            transform_steps.append(transforms.RandomCrop(spec.image_size, padding=4))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )
    return transforms.Compose(transform_steps)


def get_datasets(dataset_name: str = "cifar10", root: str | Path = "data"):
    dataset_name = dataset_name.lower()
    root = str(root)
    train_transform = build_transforms(dataset_name, train=True)
    test_transform = build_transforms(dataset_name, train=False)

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=test_transform
        )
    elif dataset_name == "svhn":
        train_dataset = torchvision.datasets.SVHN(
            root=root, split="train", download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=root, split="test", download=True, transform=test_transform
        )
    else:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")

    return train_dataset, test_dataset


def get_data_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    root: str | Path = "data",
    num_workers: int = 0,
):
    train_dataset, test_dataset = get_datasets(dataset_name=dataset_name, root=root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# Backwards-compatible alias used by older scripts.
def get_cifar10_loaders(batch_size: int = 128, root: str | Path = "data", num_workers: int = 0):
    return get_data_loaders(
        dataset_name="cifar10",
        batch_size=batch_size,
        root=root,
        num_workers=num_workers,
    )


# Για KNN / NCC

def flatten_dataset(dataset, max_samples: int | None = None):
    import numpy as np

    x_list, y_list = [], []
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(total):
        img, label = dataset[i]
        if isinstance(img, torch.Tensor):
            x_list.append(img.view(-1).cpu().numpy())
        else:
            x_list.append(torch.as_tensor(img).view(-1).cpu().numpy())
        y_list.append(int(label))

    x = np.stack(x_list, axis=0)
    y = np.array(y_list)

    return x, y


def get_class_names(dataset, dataset_name: str | None = None) -> tuple[str, ...]:
    if hasattr(dataset, "classes"):
        classes = dataset.classes
        if isinstance(classes, list):
            return tuple(str(c) for c in classes)
        return tuple(classes)

    if dataset_name is not None:
        return get_dataset_spec(dataset_name).class_names

    return tuple(str(i) for i in range(len(set(int(dataset[idx][1]) for idx in range(min(len(dataset), 1000))))))


def unnormalize_tensor(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
    spec = get_dataset_spec(dataset_name)
    mean = torch.tensor(spec.mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    std = torch.tensor(spec.std, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    return images * std + mean


def get_binary_pair_datasets(dataset_name: str, class_a: int, class_b: int, root: str = "data"):
    train_dataset, test_dataset = get_datasets(dataset_name=dataset_name, root=root)

    def filter_indices_and_relabel(dataset):
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            raise ValueError("Dataset does not expose targets or labels")

        indices = []
        for i, y in enumerate(labels):
            y = int(y)
            if dataset_name == "svhn":
                y = y % 10
            if y == class_a or y == class_b:
                indices.append(i)

        subset = Subset(dataset, indices)
        return subset

    return filter_indices_and_relabel(train_dataset), filter_indices_and_relabel(test_dataset)


def get_subset_datasets(dataset_name: str, allowed_classes: list[int], root: str = "data"):
    train_dataset, test_dataset = get_datasets(dataset_name=dataset_name, root=root)

    def filter_subset(dataset):
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            raise ValueError("Dataset does not expose targets or labels")

        indices = []
        allowed = set(allowed_classes)

        for i, y in enumerate(labels):
            y = int(y)
            if dataset_name == "svhn":
                y = y % 10
            if y in allowed:
                indices.append(i)

        return Subset(dataset, indices)

    return filter_subset(train_dataset), filter_subset(test_dataset)