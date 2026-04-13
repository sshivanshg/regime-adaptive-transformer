import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.ramt.dataset import ALL_FEATURE_COLS
    from models.ramt.encoder import MultimodalEncoder
    from models.ramt.moe import (
        ExpertTransformer,
        GatingNetwork,
        MixtureOfExperts,
        PositionalEncoding,
    )
except ModuleNotFoundError:
    # Allow direct execution: python models/ramt/model.py
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from models.ramt.dataset import ALL_FEATURE_COLS
    from models.ramt.encoder import MultimodalEncoder
    from models.ramt.moe import (
        ExpertTransformer,
        GatingNetwork,
        MixtureOfExperts,
        PositionalEncoding,
    )

# Hyperparameters and constants
TOTAL_FEATURES = len(ALL_FEATURE_COLS)
SEQUENCE_LENGTH = 30
NUM_REGIMES = 3
GROUP_DIM = 32
EMBED_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

REGIME_COL_INDEX = ALL_FEATURE_COLS.index("HMM_Regime")


class RAMTModel(nn.Module):
    """
    Wrapper model for Regime-Adaptive Multimodal Transformer (RAMT).

    Pipeline:
      1) MultimodalEncoder    : (batch, seq_len, total_features) -> (batch, seq_len, embed_dim)
      2) PositionalEncoding   : (batch, seq_len, embed_dim) -> (batch, seq_len, embed_dim)
      3) MixtureOfExperts     : experts on positional embeddings, gate conditioned on
                                 pre-positional last-step context + regime one-hot.

    Inputs:
      x:                 (batch, seq_len, total_features)
      regime_labels_one_hot:
                         one-hot (batch, num_regimes), or integer regimes
                         as (batch,) / (batch, 1) for compatibility.

    Outputs:
      prediction:        (batch, 1)
      gate_weights:      (batch, num_regimes)
    """

    def __init__(
        self,
        total_features=TOTAL_FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        num_regimes=NUM_REGIMES,
        group_dim=GROUP_DIM,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ):
        super().__init__()

        if total_features != TOTAL_FEATURES:
            raise ValueError(
                f"RAMTModel expects total_features={TOTAL_FEATURES}, got {total_features}"
            )

        self.total_features = total_features
        self.sequence_length = sequence_length
        self.num_regimes = num_regimes
        self.embed_dim = embed_dim

        self.encoder = MultimodalEncoder(
            embed_dim=embed_dim,
            group_dim=group_dim,
            dropout=dropout,
        )
        self.positional_encoding = PositionalEncoding(
            seq_len=sequence_length,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.moe = MixtureOfExperts(
            embed_dim=embed_dim,
            num_heads=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_experts=num_regimes,
            num_regimes=num_regimes,
            dropout=dropout,
        )

    def _to_one_hot(self, regime_labels):
        """Normalize regime labels to one-hot shape (batch, num_regimes)."""
        if regime_labels.dim() == 2 and regime_labels.size(-1) == self.num_regimes:
            return regime_labels.float()

        if regime_labels.dim() > 1:
            regime_labels = regime_labels.squeeze(-1)

        return F.one_hot(regime_labels.long(), num_classes=self.num_regimes).float()

    def forward(self, x, regime_labels_one_hot):
        """
        Args:
            x: (batch, seq_len, total_features=TOTAL_FEATURES)
            regime_labels_one_hot: one-hot (batch, num_regimes), or integer labels
        Returns:
            prediction: (batch, 1)
            gate_weights: (batch, num_regimes)
        """
        fused_embedding = self.encoder(x)
        positional_embedding = self.positional_encoding(fused_embedding)

        # Gate is conditioned on the pre-positional context from last timestep.
        context_last = fused_embedding[:, -1, :]
        regime_one_hot = self._to_one_hot(regime_labels_one_hot)

        prediction, gate_weights = self.moe(
            positional_embedding,
            regime_one_hot,
            gating_context=context_last,
        )
        return prediction, gate_weights


if __name__ == "__main__":
    print("Testing RAMT wrapper and components...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config constants
    batch_size = 8
    seq_len = SEQUENCE_LENGTH
    total_features = TOTAL_FEATURES
    embed_dim = EMBED_DIM
    num_regimes = NUM_REGIMES

    # 1) Positional encoding checks
    print("\n--- Test 1: PositionalEncoding ---")
    pos_enc = PositionalEncoding(seq_len=seq_len, embed_dim=embed_dim, dropout=DROPOUT).to(
        device
    )
    x_embed = torch.randn(batch_size, seq_len, embed_dim, device=device)
    x_pos = pos_enc(x_embed)
    assert x_pos.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(x_pos).any()
    print(f"Input shape:  {x_embed.shape}")
    print(f"Output shape: {x_pos.shape}")
    print("PositionalEncoding: PASSED")

    # 2) Expert forward pass checks
    print("\n--- Test 2: ExpertTransformer ---")
    expert = ExpertTransformer(
        embed_dim=embed_dim,
        num_heads=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)
    expert_out = expert(x_pos)
    assert expert_out.shape == (batch_size, 1)
    assert not torch.isnan(expert_out).any()
    print(f"Expert output shape: {expert_out.shape}")
    print("ExpertTransformer: PASSED")

    # 3) Gate normalization checks
    print("\n--- Test 3: GatingNetwork ---")
    gating = GatingNetwork(
        embed_dim=embed_dim,
        num_regimes=num_regimes,
        num_experts=num_regimes,
        hidden_dim=32,
        dropout=DROPOUT,
    ).to(device)
    context = torch.randn(batch_size, embed_dim, device=device)
    regime_idx = torch.randint(0, num_regimes, (batch_size,), device=device)
    regime_one_hot = F.one_hot(regime_idx, num_classes=num_regimes).float()

    gate_weights = gating(context, regime_one_hot)
    assert gate_weights.shape == (batch_size, num_regimes)
    assert not torch.isnan(gate_weights).any()
    sums = gate_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size, device=device), atol=1e-5)
    print(f"Gate weights shape: {gate_weights.shape}")
    print(f"Sample gate sums: {sums[:3].tolist()}")
    print("GatingNetwork: PASSED")

    # 4) Full MoE forward pass checks
    print("\n--- Test 4: MixtureOfExperts ---")
    moe = MixtureOfExperts(
        embed_dim=embed_dim,
        num_heads=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_experts=num_regimes,
        num_regimes=num_regimes,
        dropout=DROPOUT,
    ).to(device)
    moe_pred, moe_w = moe(x_pos, regime_one_hot, gating_context=context)
    assert moe_pred.shape == (batch_size, 1)
    assert moe_w.shape == (batch_size, num_regimes)
    assert not torch.isnan(moe_pred).any()
    assert not torch.isnan(moe_w).any()
    print(f"MoE prediction shape: {moe_pred.shape}")
    print(f"MoE gate shape: {moe_w.shape}")
    print("MixtureOfExperts: PASSED")

    # 5) Gradient flow checks through full RAMT wrapper
    print("\n--- Test 5: RAMT Wrapper Gradient Flow ---")
    model = RAMTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_raw = torch.randn(batch_size, seq_len, total_features, device=device)
    sampled_regimes = torch.randint(0, num_regimes, (batch_size,), device=device)
    x_raw[:, :, REGIME_COL_INDEX] = sampled_regimes.unsqueeze(1).float()
    reg_one_hot = F.one_hot(sampled_regimes, num_classes=num_regimes).float()

    pred, gw = model(x_raw, reg_one_hot)
    y_true = torch.randn(batch_size, 1, device=device)
    loss = F.mse_loss(pred, y_true)

    optimizer.zero_grad()
    loss.backward()

    grad_checks = [
        model.encoder.fusion[0].weight.grad,
        model.positional_encoding.pos_embedding.weight.grad,
        model.moe.experts[0].head[-1].weight.grad,
        model.moe.gating.gate[0].weight.grad,
    ]
    has_grad = all(g is not None and not torch.isnan(g).any() for g in grad_checks)
    assert has_grad, "Gradient flow broken in RAMT wrapper!"

    print(f"Loss: {loss.item():.6f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Gate weights shape: {gw.shape}")
    print("RAMT Wrapper Gradient Flow: PASSED")

    # Quick model summary
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 50)
    print("ALL RAMT TESTS PASSED")
    print("=" * 50)
    print("Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  sqrt(embed_dim): {math.sqrt(EMBED_DIM):.3f}")
    print(f"  Config: features={TOTAL_FEATURES}, seq_len={SEQUENCE_LENGTH}, embed={EMBED_DIM}")
    print(
        f"  Experts={NUM_REGIMES}, heads={NHEAD}, layers={NUM_LAYERS}, ff={DIM_FEEDFORWARD}"
    )
