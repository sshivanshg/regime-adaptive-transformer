"""
RAMT Loss Functions

Combined loss: MSE + Directional penalty
MSE alone optimizes magnitude accuracy.
Directional loss penalizes wrong-direction predictions.
In trading, direction matters more than magnitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Penalizes predictions with wrong sign vs actual return.

    For each sample:
      product = y_true × y_pred
      If same sign (correct direction): product > 0
      If opposite sign (wrong direction): product < 0

      loss = ReLU(-product)
      Correct direction → ReLU(negative) = 0 (no penalty)
      Wrong direction   → ReLU(positive) > 0 (penalty)

    Input:  y_pred (batch, 1), y_true (batch, 1)
    Output: scalar loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        product = y_true * y_pred
        loss = F.relu(-product)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined MSE + Directional Loss.

    total_loss = mse_loss + lambda_dir × directional_loss

    lambda_dir = 0.3:
      70% weight on magnitude (MSE)
      30% weight on direction

    Why this balance:
      Pure MSE: ignores direction completely
      Pure directional: ignores magnitude (bad for Sharpe)
      Combined: optimizes both simultaneously

    Input:  y_pred (batch, 1), y_true (batch, 1)
    Output: scalar loss, mse component, dir component
    """

    def __init__(self, lambda_dir=0.3):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.mse = nn.MSELoss()
        self.directional = DirectionalLoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        dir_loss = self.directional(y_pred, y_true)
        total = mse_loss + self.lambda_dir * dir_loss
        return total, mse_loss, dir_loss


class TournamentRankingLoss(nn.Module):
    """
    Full cross-sectional pairwise margin ranking with magnitude weighting.

    For every pair (i, j) with y_true[i] > y_true[j]:
        pair_loss = ReLU(margin - (pred[i] - pred[j])) * (y_true[i] - y_true[j])

    The (y_true[i] - y_true[j]) weight makes the tournament focus on the
    high-stakes matchups (large alpha spreads) — the trades that actually
    drive portfolio P&L. This replaces the prior top-k vs bottom-k scheme
    which dropped ~99% of the pairs and left the middle of the cross-section
    with zero gradient signal (a root cause of the pessimism bias).

    Args:
        margin: minimum required gap between pred[i] and pred[j] in UNSCALED
                alpha units (e.g. 0.02 = 2% monthly-alpha gap). Set based on
                the natural spread of the cross-section, not the RobustScaler
                transformed target.

    Input:
        pred:   (N,) or (N, 1)  — model scores for N cross-sectional items
        y_true: (N,) or (N, 1)  — true alpha, ideally in native % units

    Output: scalar loss
    """

    def __init__(self, margin: float = 0.02):
        super().__init__()
        self.margin = float(margin)

    def forward(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p = pred.reshape(-1)
        y = y_true.reshape(-1)
        N = y.shape[0]
        if N < 2:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        # Pairwise diffs: entry [i, j] = value[i] - value[j]
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)  # (N, N)
        p_diff = p.unsqueeze(1) - p.unsqueeze(0)  # (N, N)

        # Keep only ordered pairs where y[i] > y[j] (avoid double-counting + self pairs)
        mask = (y_diff > 0).float()
        weight = y_diff.abs() * mask               # magnitude weight
        hinge = F.relu(self.margin - p_diff)       # penalize pred gaps < margin

        num = (hinge * weight).sum()
        den = weight.sum() + 1e-8
        return num / den
