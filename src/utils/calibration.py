"""Post-hoc calibration utilities: temperature scaling and Expected Calibration Error.

Temperature scaling divides the logits by a single scalar T (learned on a held-out
validation set) before the softmax/sigmoid. Because T is shared across classes it does
not change the argmax, so accuracy is unchanged; only the confidence values are rescaled
to better match the true probability of being correct.
"""

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Wraps a single temperature applied to logits: logits -> logits / T."""

    def __init__(self):
        super().__init__()
        # Optimise log_temperature so the temperature stays strictly positive.
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, mode: str = "multiclass",
                    max_iter: int = 100, lr: float = 0.01) -> float:
    """Find the temperature that minimises NLL on (logits, labels).

    mode='multiclass': logits (N, C), labels (N,) int   -> CrossEntropyLoss
    mode='binary':     logits (N,)    , labels (N,) {0,1} -> BCEWithLogitsLoss
    Returns the scalar temperature.
    """
    logits = logits.detach()
    labels = labels.detach()
    scaler = TemperatureScaler()

    if mode == "multiclass":
        criterion = nn.CrossEntropyLoss()
        target = labels.long()
    elif mode == "binary":
        criterion = nn.BCEWithLogitsLoss()
        target = labels.float()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature


def _ece_from_confidences(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error given per-sample confidence and correctness."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def ece_multiclass(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE for a softmax classifier. probs (N, C), labels (N,)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels)
    return _ece_from_confidences(confidences, correct, n_bins)


def ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE for a binary classifier. probs (N,) = sigmoid output, labels (N,) in {0,1}."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1.0 - probs)
    correct = (preds == labels)
    return _ece_from_confidences(confidences, correct, n_bins)
