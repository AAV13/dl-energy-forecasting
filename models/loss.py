"""
Quantile Huber (Pinball Huber) loss for GRQ-PatchTST.

Combines pinball loss with Huber smoothing to handle the heavy-tailed
electricity price distribution (-90 to +200 EUR/MWh) without exploding
gradients during extreme events.
"""

import torch
import torch.nn as nn
from typing import List


QUANTILES = [0.1, 0.5, 0.9]


def pinball_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: List[float] = QUANTILES,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Pinball Huber loss for a single target.

    Args:
        pred:      (B, horizon, n_quantiles)
        target:    (B, horizon)
        quantiles: quantile levels e.g. [0.1, 0.5, 0.9]
        delta:     Huber smoothing threshold

    Returns:
        scalar loss
    """
    # Expand target to match pred
    tgt = target.unsqueeze(-1).expand_as(pred)   # (B, horizon, Q)
    errors = tgt - pred                           # positive = under-prediction

    total = torch.tensor(0.0, device=pred.device)

    for i, q in enumerate(quantiles):
        e = errors[..., i]                        # (B, horizon)
        abs_e = e.abs()

        # Huber smoothing
        huber = torch.where(
            abs_e <= delta,
            0.5 * e.pow(2),
            delta * (abs_e - 0.5 * delta),
        )

        # Asymmetric pinball weighting
        pinball = torch.where(e >= 0, q * huber, (q - 1.0) * huber)
        total = total + pinball.mean()

    return total / len(quantiles)


def grq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_weights: List[float] = [1.0, 1.0, 2.0],
    quantiles: List[float] = QUANTILES,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Weighted Quantile Huber loss across all targets.

    Args:
        pred:           (B, horizon, n_targets, n_quantiles)
        target:         (B, n_targets, horizon)  ← standard Y_batch shape
        target_weights: per-target economic weights [solar, wind, price]
        quantiles:      quantile levels
        delta:          Huber threshold

    Returns:
        scalar weighted loss
    """
    # target: (B, n_targets, horizon) → (B, horizon, n_targets)
    tgt = target.permute(0, 2, 1)

    total = torch.tensor(0.0, device=pred.device)

    for i, w in enumerate(target_weights):
        loss_i = pinball_huber(
            pred[:, :, i, :],   # (B, horizon, Q)
            tgt[:, :, i],       # (B, horizon)
            quantiles=quantiles,
            delta=delta,
        )
        total = total + w * loss_i

    return total / sum(target_weights)