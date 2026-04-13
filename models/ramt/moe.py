import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Transformer.
    Adds position information to sequence embeddings.
    Without this, Transformer cannot distinguish
    Day 1 from Day 30 - they look identical.

    Uses learnable embeddings (not fixed sinusoidal)
    because financial time patterns are data-driven
    not mathematically fixed.

    Input:  (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(self, seq_len=30, embed_dim=64, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        if seq_len > self.pos_embedding.num_embeddings:
            raise ValueError(
                "Input sequence length exceeds positional embedding capacity: "
                f"got seq_len={seq_len}, max_supported={self.pos_embedding.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size, -1
        )
        # positions: (batch, seq_len)
        pos_enc = self.pos_embedding(positions)
        # pos_enc: (batch, seq_len, embed_dim)
        return self.dropout(x + pos_enc)


class ExpertTransformer(nn.Module):
    """
    A single regime-specialized Transformer expert.
    Each expert has identical architecture but
    completely separate learned weights.

    This allows each expert to specialize:
    - Bull expert: learns momentum continuation patterns
    - Bear expert: learns reversal and risk patterns
    - HighVol expert: learns volatility spike patterns

    Architecture:
        TransformerEncoder (2 layers, 4 heads)
        Pool last timestep
        LayerNorm
        Linear(embed_dim -> hidden_dim) -> ReLU -> Dropout
        Linear(hidden_dim -> 1)

    Input:  (batch, seq_len, embed_dim)
    Output: (batch, 1) - return prediction
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, features)
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        hidden_dim = embed_dim // 2  # 32
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        out = self.transformer(x)
        # out: (batch, seq_len, embed_dim)

        # Take last timestep only
        # Day 30 contains most recent information
        # and attends to all previous days via attention
        last = out[:, -1, :]
        # last: (batch, embed_dim)

        return self.head(last)
        # output: (batch, 1)


class GatingNetwork(nn.Module):
    """
    Computes soft routing weights over N experts.

    Takes two inputs:
    1. Context embedding from last timestep
       (what does the market look like now?)
    2. Regime one-hot encoding
       (what regime does HMM say we are in?)

    Combines both to produce expert weights.

    Why both context AND regime:
        Regime alone: "It is bear regime"
        Context alone: "Volatility is elevated, momentum negative"
        Together: "It is bear regime AND here is exactly
                   what the market looks like" -> better routing

    Input:  context (batch, embed_dim)
            regime (batch,) integers 0/1/2 OR one-hot (batch, num_regimes)
    Output: (batch, num_experts) softmax weights
    """

    def __init__(
        self,
        embed_dim=64,
        num_regimes=3,
        num_experts=3,
        hidden_dim=32,
        dropout=0.1,
    ):
        super().__init__()

        # Input: embed_dim + num_regimes (one-hot)
        input_dim = embed_dim + num_regimes

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            # No softmax here - applied in forward
            # to allow temperature scaling later
        )

        self.num_regimes = num_regimes
        self.num_experts = num_experts

    def forward(self, context, regime):
        """
        Args:
            context: (batch, embed_dim)
            regime:  (batch,) integer tensor, values 0/1/2,
                     (batch, 1), or one-hot (batch, num_regimes)
        Returns:
            weights: (batch, num_experts) summing to 1
        """
        # Support one-hot regime directly for wrapper-level routing,
        # while keeping compatibility with integer regime labels.
        if regime.dim() == 2 and regime.size(-1) == self.num_regimes:
            regime_onehot = regime.float()
        else:
            # DataLoader commonly provides regime as (batch, 1); flatten to (batch,)
            if regime.dim() > 1:
                regime = regime.squeeze(-1)

            # Convert regime integers to one-hot
            # regime 0 -> [1, 0, 0]
            # regime 1 -> [0, 1, 0]
            # regime 2 -> [0, 0, 1]
            regime_onehot = F.one_hot(regime.long(), num_classes=self.num_regimes).float()
        # regime_onehot: (batch, num_regimes)
        regime_onehot = regime_onehot.to(context.device)

        # Concatenate context with regime
        gate_input = torch.cat([context, regime_onehot], dim=-1)
        # gate_input: (batch, embed_dim + num_regimes)

        # Compute logits then softmax
        logits = self.gate(gate_input)
        weights = F.softmax(logits, dim=-1)
        # weights: (batch, num_experts) - sum to 1.0

        return weights


class MixtureOfExperts(nn.Module):
    """
    Combines N ExpertTransformers with GatingNetwork routing.

    This is the core novelty of RAMT vs plain LSTM.

    LSTM: one model -> one prediction (ignores regime)
    MoE:  three specialists -> weighted blend (regime-aware)

    Forward pass:
    1. Run ALL three experts independently on same input
    2. Get gate weights from GatingNetwork
    3. Blend expert predictions using gate weights

    Example:
        Bull expert predicts:    +0.8%
        Bear expert predicts:    -0.3%
        HighVol expert predicts: +0.1%

        Gate weights (bear regime):
        Bull=0.10, Bear=0.75, HighVol=0.15

        Final prediction:
        0.10x(+0.8%) + 0.75x(-0.3%) + 0.15x(+0.1%)
        = +0.08% - 0.225% + 0.015%
        = -0.13% (negative, bear dominates)

    Input:  x (batch, seq_len, embed_dim)
            regime (batch,) integers 0/1/2
    Output: prediction (batch, 1)
            gate_weights (batch, num_experts) for analysis
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        num_experts=3,
        num_regimes=3,
        dropout=0.1,
    ):
        super().__init__()

        self.num_experts = num_experts

        # Create N independent expert Transformers
        # Each has identical architecture but separate weights
        self.experts = nn.ModuleList(
            [
                ExpertTransformer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gating = GatingNetwork(
            embed_dim=embed_dim,
            num_regimes=num_regimes,
            num_experts=num_experts,
            hidden_dim=32,
            dropout=dropout,
        )

    def forward(self, x, regime, gating_context=None):
        """
        Args:
            x:      (batch, seq_len, embed_dim)
            regime: (batch,) integer labels, (batch, 1),
                    or one-hot (batch, num_regimes)
            gating_context: optional (batch, embed_dim) context override
                            for gating. If None, uses x[:, -1, :].
        Returns:
            prediction:   (batch, 1)
            gate_weights: (batch, num_experts)
        """
        # Get context from last timestep for gating
        context = x[:, -1, :] if gating_context is None else gating_context
        # context: (batch, embed_dim)

        # Compute gate weights
        gate_weights = self.gating(context, regime)
        # gate_weights: (batch, num_experts)

        # Run all experts independently
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)
            # out: (batch, 1)
            expert_outputs.append(out)

        # Stack expert outputs
        expert_stack = torch.stack(expert_outputs, dim=1)
        # expert_stack: (batch, num_experts, 1)

        # Weighted blend using gate weights
        # gate_weights: (batch, num_experts)
        # Unsqueeze for broadcasting: (batch, num_experts, 1)
        weights = gate_weights.unsqueeze(-1)

        # Weighted sum across experts
        prediction = (weights * expert_stack).sum(dim=1)
        # prediction: (batch, 1)

        return prediction, gate_weights


if __name__ == "__main__":
    print("Testing MoE components...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    batch_size = 8
    seq_len = 30
    embed_dim = 64
    num_experts = 3

    # -- Test 1: PositionalEncoding --
    print("\n--- Test 1: PositionalEncoding ---")
    pos_enc = PositionalEncoding(seq_len=seq_len, embed_dim=embed_dim).to(device)

    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    out = pos_enc(x)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(out).any()
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("PositionalEncoding: PASSED")

    # -- Test 2: ExpertTransformer --
    print("\n--- Test 2: ExpertTransformer ---")
    expert = ExpertTransformer(embed_dim=embed_dim, num_heads=4, num_layers=2).to(
        device
    )

    expert_params = sum(p.numel() for p in expert.parameters())
    print(f"Expert parameters: {expert_params:,}")

    out = expert(x)
    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Sample predictions: {out[:4].squeeze().tolist()}")
    print("ExpertTransformer: PASSED")

    # -- Test 3: GatingNetwork --
    print("\n--- Test 3: GatingNetwork ---")
    gating = GatingNetwork(embed_dim=embed_dim, num_regimes=3, num_experts=num_experts).to(
        device
    )

    context = torch.randn(batch_size, embed_dim).to(device)
    regime = torch.randint(0, 3, (batch_size,)).to(device)

    weights = gating(context, regime)
    assert weights.shape == (batch_size, num_experts)
    assert not torch.isnan(weights).any()

    # Also support DataLoader-style regime shape: (batch, 1)
    weights_2d = gating(context, regime.unsqueeze(-1))
    assert weights_2d.shape == (batch_size, num_experts)
    assert not torch.isnan(weights_2d).any()

    # Verify weights sum to 1
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(batch_size, device=device), atol=1e-5), (
        "Gate weights do not sum to 1!"
    )

    print(f"Context shape: {context.shape}")
    print(f"Regime values: {regime.tolist()}")
    print(f"Gate weights shape: {weights.shape}")
    print("Gate weights (first 3 samples):")
    for i in range(3):
        w = weights[i].tolist()
        r = regime[i].item()
        regime_name = ["HighVol", "Bull", "Bear"][r]
        print(
            f"  Sample {i} (regime={regime_name}): "
            f"HV={w[0]:.3f} Bull={w[1]:.3f} Bear={w[2]:.3f}"
        )
    print(f"Weight sums: {weight_sums.tolist()[:3]}")
    print("GatingNetwork: PASSED")

    # -- Test 4: Full MixtureOfExperts --
    print("\n--- Test 4: MixtureOfExperts ---")
    moe = MixtureOfExperts(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        num_experts=num_experts,
    ).to(device)

    total_params = sum(p.numel() for p in moe.parameters())
    print(f"MoE total parameters: {total_params:,}")

    x_seq = torch.randn(batch_size, seq_len, embed_dim).to(device)
    regime_batch = torch.randint(0, 3, (batch_size,)).to(device)

    prediction, gate_weights = moe(x_seq, regime_batch)

    assert prediction.shape == (batch_size, 1)
    assert gate_weights.shape == (batch_size, num_experts)
    assert not torch.isnan(prediction).any()
    assert not torch.isnan(gate_weights).any()

    # Also support DataLoader-style regime shape: (batch, 1)
    prediction_2d, gate_weights_2d = moe(x_seq, regime_batch.unsqueeze(-1))
    assert prediction_2d.shape == (batch_size, 1)
    assert gate_weights_2d.shape == (batch_size, num_experts)

    print(f"Input shape:       {x_seq.shape}")
    print(f"Prediction shape:  {prediction.shape}")
    print(f"Gate weights shape:{gate_weights.shape}")
    print(f"Sample predictions: {prediction[:4].squeeze().tolist()}")
    print("Sample gate weights:")
    for i in range(3):
        w = gate_weights[i].tolist()
        r = regime_batch[i].item()
        regime_name = ["HighVol", "Bull", "Bear"][r]
        print(
            f"  regime={regime_name}: "
            f"HV={w[0]:.3f} Bull={w[1]:.3f} Bear={w[2]:.3f}"
        )
    print("MixtureOfExperts: PASSED")

    # -- Test 5: Gradient Flow --
    print("\n--- Test 5: Gradient Flow ---")
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

    pred, weights = moe(x_seq, regime_batch)
    y_true = torch.randn(batch_size, 1).to(device)
    loss = F.mse_loss(pred, y_true)
    loss.backward()

    # Check gradients exist and are not NaN
    has_grad = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in moe.parameters()
        if p.requires_grad
    )
    assert has_grad, "Gradient flow broken!"
    print(f"Loss: {loss.item():.6f}")
    print("Gradients: flowing correctly")
    print("Gradient Flow: PASSED")

    print("\n" + "=" * 50)
    print("ALL MOE TESTS PASSED")
    print("=" * 50)
    print("\nSummary:")
    print(f"  PositionalEncoding params: {sum(p.numel() for p in pos_enc.parameters()):,}")
    print(f"  Single Expert params: {expert_params:,}")
    print(f"  Full MoE params: {total_params:,}")
    print("  (3 experts + gating network)")