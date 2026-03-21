from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEHead(nn.Module):
    """
    Top-1 sparse MoE: gate selects one expert; auxiliary load-balancing loss optional.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_hidden: int,
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z: (B, D) pooled representation
        Returns logits (B,) or (B, num_classes), and gate probs (B, E) for aux loss.
        """
        g = F.softmax(self.gate(z), dim=-1)
        # top-1 routing
        idx = g.argmax(dim=-1)
        logits_list = []
        for b in range(z.size(0)):
            e = int(idx[b].item())
            logits_list.append(self.experts[e](z[b : b + 1]).squeeze(-1))
        logits = torch.cat(logits_list, dim=0)
        return logits, g


def moe_load_balance_loss(g: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Encourage uniform expert usage: CV^2 style penalty on mean gate mass per expert."""
    # g: (B, E)
    fr = g.mean(dim=0)
    target = 1.0 / num_experts
    return ((fr - target) ** 2).sum()
