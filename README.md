# neural-network-system2

Cleaned and reorganized neural-network project for a thesis on baseline classification and System 2-inspired extensions.

## Supported datasets

The codebase now supports:
- CIFAR-10
- CIFAR-100
- SVHN

## Repository structure

```text
neural-network-system2/
├── src/
│   ├── models/          # CNN architecture
│   ├── pipelines/       # baseline and future System 2 pipeline
│   ├── training/        # training entry points
│   ├── evaluation/      # evaluation scripts
│   └── utils/           # data, metrics, TTA, kNN helpers
├── experiments/         # exploratory scripts and comparisons
├── docs/                # notes
├── data/                # dataset download location
├── artifacts/           # saved weights (ignored by git)
└── results/             # generated outputs (ignored by git)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train baseline CNN

```bash
python -m src.training.train_cnn --dataset cifar10 --epochs 20 --output artifacts/cnn_cifar10.pth
python -m src.training.train_cnn --dataset cifar100 --epochs 20 --output artifacts/cnn_cifar100.pth
python -m src.training.train_cnn --dataset svhn --epochs 20 --output artifacts/cnn_svhn.pth
```

## Evaluate with TTA

```bash
python -m src.evaluation.evaluate_tta --dataset cifar10 --weights artifacts/cnn_cifar10.pth
python -m src.evaluation.evaluate_tta --dataset cifar100 --weights artifacts/cnn_cifar100.pth
python -m src.evaluation.evaluate_tta --dataset svhn --weights artifacts/cnn_svhn.pth
```

## Save example predictions

```bash
python -m src.pipelines.save_examples --dataset cifar10 --weights artifacts/cnn_cifar10.pth
```

## Experiments

```bash
python -m experiments.run_experiments --dataset cifar10
python -m experiments.run_knn_ncc --dataset cifar100
python -m experiments.evaluate_knn_confidence --dataset svhn --weights artifacts/cnn_svhn.pth
```