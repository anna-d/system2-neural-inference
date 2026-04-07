# neural-network-system2

Cleaned and reorganized CIFAR-10 project for a thesis on baseline neural inference and System 2-inspired extensions.

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
python -m src.training.train_cnn --epochs 20 --output artifacts/cnn_cifar10.pth
```

## Evaluate with TTA

```bash
python -m src.evaluation.evaluate_tta --weights artifacts/cnn_cifar10.pth
```

## Save example predictions

```bash
python -m src.pipelines.save_examples --weights artifacts/cnn_cifar10.pth
```

## Experiments

```bash
python experiments/run_experiments.py
python experiments/run_knn_ncc.py
python experiments/evaluate_knn_confidence.py --weights artifacts/cnn_cifar10.pth
```
