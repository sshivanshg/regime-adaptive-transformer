from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import TransformerEncoder
from .moe import MoEHead, moe_load_balance_loss


class RegimeAdaptiveTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        num_experts: int = 4,
        expert_hidden: int = 128,
        dropout: float = 0.1,
        moe_balance_coef: float = 0.01,
    ):
        super().__init__()
        self.moe_balance_coef = moe_balance_coef
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.pool = nn.LayerNorm(d_model)
        self.moe = MoEHead(
            d_model=d_model,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
            num_classes=1,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        h = self.encoder(x)
        z = self.pool(h[:, -1])
        logits, g = self.moe(z)
        aux = (
            self.moe_balance_coef * moe_load_balance_loss(g, self.moe.num_experts)
            if self.training
            else None
        )
        return logits, g, aux


def bce_with_logits_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    aux: torch.Tensor | None,
) -> torch.Tensor:
    yf = y.float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, yf)
    if aux is not None:
        loss = loss + aux
    return loss
