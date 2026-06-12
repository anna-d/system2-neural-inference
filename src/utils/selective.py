"""Helpers for budget-aware (selective) triggering.

A budget tau means System 2 runs on at most tau fraction of inputs: E[T(x)] <= tau.
Instead of a fixed probability threshold, we pick the threshold so that the most
uncertain tau fraction of the data triggers. The trigger score is one where LOWER
means MORE uncertain (max-softmax confidence, or top1-top2 margin), and the trigger
fires when score < threshold, so threshold = the tau-quantile of the scores.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def budget_threshold(model, loader, device, budget, mode="confidence", temperature=1.0):
    """Return the trigger threshold that makes ~`budget` fraction of inputs trigger.

    mode='confidence': score = max softmax probability
    mode='margin':     score = top1 - top2 softmax probability
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError("--budget must be in (0, 1].")

    scores = []
    for batch in loader:
        images = batch[0].to(device)
        probs = F.softmax(model(images) / temperature, dim=1)
        if mode == "confidence":
            s = probs.max(dim=1).values
        elif mode == "margin":
            top2 = probs.topk(2, dim=1).values
            s = top2[:, 0] - top2[:, 1]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        scores.append(s.cpu())

    scores = torch.cat(scores)
    return float(torch.quantile(scores, budget).item())
