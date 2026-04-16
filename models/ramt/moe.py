import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplainableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Drop-in replacement for nn.TransformerEncoderLayer that stores attention weights.

    Important: We keep the exact same parameter structure as PyTorch's built-in
    TransformerEncoderLayer, so state_dict keys remain compatible.

    After a forward pass, the most recent self-attention weights are available at:
      self.last_attn_weights  # shape: (batch, heads, tgt_len, src_len)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal: bool = False):
        # Request per-head weights (average_attn_weights=False)
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.last_attn_weights = attn_weights
        return self.dropout1(attn_output)


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
        Return the last-timestep embedding (batch, embed_dim)

    Input:  (batch, seq_len, embed_dim)
    Output: (batch, embed_dim) - shared representation
    """

    def __init__(
        self,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        explainable_attn: bool = False,
    ):
        super().__init__()

        layer_cls = ExplainableTransformerEncoderLayer if explainable_attn else nn.TransformerEncoderLayer
        encoder_layer = layer_cls(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, features)
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.explainable_attn = explainable_attn

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        out = self.transformer(x)
        # out: (batch, seq_len, embed_dim)

        # Take last timestep only
        # Day 30 contains most recent information
        # and attends to all previous days via attention
        last = out[:, -1, :]
        # last: (batch, embed_dim)

        return last
        # output: (batch, embed_dim)

    def get_last_attn_stack(self) -> list[torch.Tensor]:
        """
        Return attention weights from each encoder layer (if explainable_attn=True).
        Each element is a tensor with shape (batch, heads, tgt_len, src_len).
        """
        if not self.explainable_attn:
            return []
        layers = getattr(self.transformer, "layers", None)
        if layers is None:
            return []
        out: list[torch.Tensor] = []
        for lyr in layers:
            w = getattr(lyr, "last_attn_weights", None)
            if w is not None:
                out.append(w)
        return out


class RegimeCrossAttention(nn.Module):
    """
    Regime-as-Query cross-attention block.

    Regime acts as the Query; the fused feature sequence acts as Key/Value.
    This lets the HMM regime dynamically reweight WHICH features matter
    at each timestep, instead of being treated as just another modality
    concatenated in the encoder.

    Intuition:
        - In BULL regime, the learned prototype attends to momentum features
          (Ret_21d, Volume_Surge).
        - In BEAR, it attends to mean-reversion signals (RSI_14, BB_Dist).
        - In HIGH_VOL, it attends to macro risk proxies (INDIAVIX, USDINR).

    Input:
        features: (B, T, D) post-encoder, pre-MoE sequence embedding
        regime:   (B,) integer regime label, or (B, 1)

    Output:
        (B, T, D) regime-contextualized sequence (residual + FFN block).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_regimes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_regimes = int(num_regimes)
        self.regime_query = nn.Embedding(self.num_regimes, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        B, T, D = features.shape
        # Normalize regime to integer (B,)
        if regime.dim() > 1:
            regime = regime.squeeze(-1)
        q = self.regime_query(regime.long())        # (B, D)
        q = q.unsqueeze(1).expand(B, T, D)           # (B, T, D)
        ctx, _ = self.cross_attn(query=q, key=features, value=features)
        x = self.norm(features + ctx)                # residual 1 (attention)
        x = self.norm2(x + self.ffn(x))              # residual 2 (FFN)
        return x


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
    Output: fused_context (batch, embed_dim)
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
        explainable_attn: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.explainable_attn = explainable_attn

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
                    explainable_attn=explainable_attn,
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
            fused_context: (batch, embed_dim)
            gate_weights: (batch, num_experts)
        """
        # Get context from last timestep for gating
        context = x[:, -1, :] if gating_context is None else gating_context
        # context: (batch, embed_dim)

        # Compute gate weights
        gate_weights = self.gating(context, regime)
        # gate_weights: (batch, num_experts)

        # Run all experts independently (return context embeddings)
        expert_outputs: list[torch.Tensor] = []
        for expert in self.experts:
            out = expert(x)
            # out: (batch, embed_dim)
            expert_outputs.append(out)

        # Stack expert outputs
        expert_stack = torch.stack(expert_outputs, dim=1)
        # expert_stack: (batch, num_experts, embed_dim)

        # Weighted blend using gate weights
        # gate_weights: (batch, num_experts)
        # Unsqueeze for broadcasting: (batch, num_experts, 1)
        weights = gate_weights.unsqueeze(-1)

        # Weighted sum across experts -> fused context
        fused_context = (weights * expert_stack).sum(dim=1)
        # fused_context: (batch, embed_dim)

        return fused_context, gate_weights

    def get_last_attention(self) -> list[list[torch.Tensor]]:
        """
        Returns per-expert attention stacks (list of layers) from the last forward pass.
        Shape per tensor: (batch, heads, tgt_len, src_len).
        """
        if not self.explainable_attn:
            return []
        out: list[list[torch.Tensor]] = []
        for e in self.experts:
            out.append(e.get_last_attn_stack())
        return out


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