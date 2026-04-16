"""
RAMT — Full Model Architecture

Complete forward pass:
1. MultimodalEncoder  → (batch, seq_len, 64)
2. PositionalEncoding → (batch, seq_len, 64)
3. MixtureOfExperts   → (batch, embed_dim) context embedding
                        (batch, 3) gate weights

Regime conditioning flows through:
- RegimeEncoder in MultimodalEncoder
- GatingNetwork in MixtureOfExperts
Both use the same HMM_Regime signal.
"""

import torch
import torch.nn as nn

from models.ramt.dataset import ALL_FEATURE_COLS
from models.ramt.encoder import MultimodalEncoder
from models.ramt.moe import MixtureOfExperts, PositionalEncoding, RegimeCrossAttention


class RAMTModel(nn.Module):
    """
    Regime-Adaptive Multimodal Transformer.

    Architecture summary:

    Input: (batch, seq_len=30, num_features=len(ALL_FEATURE_COLS))
           + regime (batch,) integer 0/1/2

    Step 1 — MultimodalEncoder:
      Split features into price / tech / volume / macro + regime + ticker
      Encode each group separately
      Fuse into (batch, seq_len, embed_dim=64)

    Step 2 — PositionalEncoding:
      Add learnable position information
      Day 1 vs Day 30 become distinguishable
      Still (batch, seq_len, embed_dim=64)

    Step 3 — MixtureOfExperts:
      3 Transformer experts process the sequence
      GatingNetwork blends based on regime
      Output: (batch, embed_dim) context embedding
              (batch, 3) gate weights

    Step 4 — Dual heads ("dual-brain"):
      Monthly head: predict Monthly_Alpha / Monthly_Alpha_Z (ranking signal)
      Daily head:   predict next-day Log_Return (sanity-check signal)

    Output: prediction (batch, 1)
            gate_weights (batch, 3) — for interpretability

    Args:
        embed_dim: unified embedding dimension (default 64)
        group_dim: per-group encoder output dim (default 32)
        num_heads: attention heads in Transformer (default 4)
        num_transformer_layers: layers per expert (default 2)
        dim_feedforward: feedforward size in Transformer (default 128)
        num_experts: number of MoE experts (default 3)
        num_regimes: number of HMM regimes (default 3)
        seq_len: input sequence length (default 30)
        dropout: dropout rate (default 0.1)
    """

    def __init__(
        self,
        embed_dim=64,
        group_dim=32,
        num_heads=4,
        num_transformer_layers=2,
        dim_feedforward=128,
        num_experts=3,
        num_regimes=3,
        seq_len=30,
        dropout=0.1,
        explainable_attn: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Step 1: Multimodal encoder
        self.encoder = MultimodalEncoder(
            embed_dim=embed_dim,
            group_dim=group_dim,
            dropout=dropout,
        )

        # Step 2: Positional encoding
        self.pos_encoding = PositionalEncoding(
            seq_len=seq_len,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.regime_attn = RegimeCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_regimes=num_regimes,
            dropout=dropout,
        )

        # Step 3: Mixture of Experts (returns fused context embedding)
        self.moe = MixtureOfExperts(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            num_regimes=num_regimes,
            dropout=dropout,
            explainable_attn=explainable_attn,
        )

        # Step 4: Dual heads
        head_hidden = embed_dim // 2
        self.monthly_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.daily_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x, regime, ticker_id=None):
        """
        Full RAMT forward pass.

        Args:
            x:      (batch, seq_len, F) scaled feature tensor (F = len(ALL_FEATURE_COLS))
            regime: (batch,) integer regime labels 0/1/2

        Returns:
            prediction:   (batch, 1) next-day return prediction
            gate_weights: (batch, num_experts) for analysis
        """
        # Step 1: Encode each feature group separately
        # then fuse into unified representation
        encoded = self.encoder(x, regime, ticker_id=ticker_id)
        # encoded: (batch, seq_len, embed_dim=64)

        # Step 2: Add positional information
        # So Transformer knows Day 1 from Day 30
        positioned = self.pos_encoding(encoded)
        # positioned: (batch, seq_len, embed_dim=64)

        positioned = self.regime_attn(positioned, regime)

        # Step 3: Route through regime-specialized experts -> context embedding
        context, gate_weights = self.moe(positioned, regime)
        # context: (batch, embed_dim)
        # gate_weights: (batch, num_experts)

        pred_monthly = self.monthly_head(context)
        pred_daily = self.daily_head(context)

        return pred_monthly, pred_daily, gate_weights

    def count_parameters(self):
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ramt(config=None):
    """
    Factory function to build RAMTModel with config dict.
    Uses defaults from PHASE2_PLAN.md if no config provided.

    Default config:
        embed_dim=64, group_dim=32, num_heads=4,
        num_transformer_layers=2, dim_feedforward=128,
        num_experts=3, num_regimes=3, seq_len=30, dropout=0.1
    """
    defaults = {
        "embed_dim": 64,
        "group_dim": 32,
        "num_heads": 4,
        "num_transformer_layers": 2,
        "dim_feedforward": 128,
        "num_experts": 3,
        "num_regimes": 3,
        "seq_len": 30,
        "dropout": 0.1,
        "explainable_attn": False,
    }
    if config:
        defaults.update(config)
    return RAMTModel(**defaults)


if __name__ == "__main__":
    print("Testing Full RAMT Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_ramt().to(device)
    total_params = model.count_parameters()
    print(f"Total RAMT parameters: {total_params:,}")

    # Parameter breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    pos_params = sum(p.numel() for p in model.pos_encoding.parameters())
    moe_params = sum(p.numel() for p in model.moe.parameters())
    print(f"  Encoder:            {enc_params:,}")
    print(f"  PositionalEncoding: {pos_params:,}")
    print(f"  MoE:                {moe_params:,}")

    # Test forward pass with random data
    print("\n--- Test 1: Forward Pass (random data) ---")
    batch_size = 16
    seq_len = 30
    num_features = len(ALL_FEATURE_COLS)

    x = torch.randn(batch_size, seq_len, num_features).to(device)
    regime = torch.randint(0, 3, (batch_size,)).to(device)

    pred_m, pred_d, weights = model(x, regime)

    assert pred_m.shape == (batch_size, 1), (
        f"Expected ({batch_size}, 1) got {pred_m.shape}"
    )
    assert pred_d.shape == (batch_size, 1), (
        f"Expected ({batch_size}, 1) got {pred_d.shape}"
    )
    assert weights.shape == (batch_size, 3), (
        f"Expected ({batch_size}, 3) got {weights.shape}"
    )
    assert not torch.isnan(pred_m).any(), "NaN in predictions!"
    assert not torch.isnan(weights).any(), "NaN in gate weights!"

    print(f"Input shape:       {x.shape}")
    print(f"Monthly pred shape: {pred_m.shape}")
    print(f"Gate weights shape:{weights.shape}")
    print(f"Predictions range: [{pred_m.min():.4f}, {pred_m.max():.4f}]")
    print("Forward Pass: PASSED")

    # Test with real data
    print("\n--- Test 2: Forward Pass (real data) ---")
    from models.ramt.dataset import RAMTDataModule

    dm = RAMTDataModule("JPM", seq_len=30, batch_size=16)
    folds = dm.get_walk_forward_indices()
    train_idx, test_idx = folds[0]
    train_loader, val_loader, test_loader, dates = dm.get_fold_loaders(
        train_idx, test_idx
    )

    X_batch, y_batch, r_batch = next(iter(train_loader))
    X_batch = X_batch.to(device)
    r_batch = r_batch.squeeze(-1).to(device)

    pred_m_real, _pred_d_real, weights_real = model(X_batch, r_batch)

    assert not torch.isnan(pred_m_real).any()
    print(f"Real input shape:  {X_batch.shape}")
    print(f"Real pred shape:   {pred_m_real.shape}")
    print(f"Regime values:     {r_batch.tolist()[:8]}")
    print("Gate weights sample:")
    for i in range(3):
        w = weights_real[i].tolist()
        r = r_batch[i].item()
        name = ["HighVol", "Bull", "Bear"][r]
        print(
            f"  {name}: HV={w[0]:.3f} "
            f"Bull={w[1]:.3f} Bear={w[2]:.3f}"
        )
    print("Real Data Test: PASSED")

    # Test loss function
    print("\n--- Test 3: Loss Function ---")
    from models.ramt.losses import CombinedLoss

    criterion = CombinedLoss(lambda_dir=0.3)
    y_true = y_batch.to(device)

    total_loss, mse_loss, dir_loss = criterion(pred_m_real, y_true)

    assert not torch.isnan(total_loss)
    print(f"Total loss:       {total_loss.item():.6f}")
    print(f"MSE loss:         {mse_loss.item():.6f}")
    print(f"Directional loss: {dir_loss.item():.6f}")
    print("Loss Function: PASSED")

    # Test backward pass
    print("\n--- Test 4: Backward Pass ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    optimizer.zero_grad()
    pred_bp, _pd, _gw = model(X_batch, r_batch)
    loss, _, _ = criterion(pred_bp, y_true)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    has_grad = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    assert has_grad, "Some parameters have no gradient!"
    print(f"Loss before step: {loss.item():.6f}")
    print("All gradients:    flowing")
    print("Optimizer step:   completed")
    print("Backward Pass: PASSED")

    print("\n" + "=" * 50)
    print("ALL RAMT MODEL TESTS PASSED")
    print("=" * 50)
    print("\nModel is ready for training.")
    print(f"Total parameters: {total_params:,}")
    print("Next step: python models/ramt/train.py")
