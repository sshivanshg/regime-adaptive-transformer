# Triple-Expert Hybrid System Architecture

This diagram illustrates the **Foundation-Hybrid** architecture implemented in Phase 3. It showcases the synergy between the Technical, Foundation, and Risk experts, modulated by a Regime-Adaptive Fusion mechanism.

```mermaid
graph TD
    subgraph "Data Input Layer"
        D[NIFTY 200 Features] --> |10 RAMT Features| F[Feature Engineering]
        F --> |Technical| M[Momentum Expert]
        F --> |Time-Series Sequences| C[Foundation Expert]
        F --> |Market Indices| R[Risk Expert]
    end

    subgraph "Expert Layer"
        M --> |Signal 1| S1[Ret_21d Momentum]
        
        subgraph "Chronos-T5 + LoRA"
            C --> |Encoder Inputs| T5[Chronos-T5 Backbone]
            T5 --> |Attention Maps| LORA[LoRA Adapters: W = W₀ + BA]
            LORA --> |Rank=8| RH[Ranking Head]
            RH --> |Signal 2| S2[Chronos Alpha Score]
        end
        
        R --> |HMM Detection| HMM[Regime Expert]
        HMM --> |State| RG[Regime: Bull / Vol / Bear]
    end

    subgraph "Adaptive Fusion Layer"
        RG --> |Modulate Weights| AF[Regime-Adaptive Fusion]
        S1 --> AF
        S2 --> AF
        
        AF --> |Bull| B[70% Mom / 30% Chronos]
        AF --> |Volatile| V[30% Mom / 70% Chronos]
        AF --> |Bear| BE[10% Mom / 90% Chronos]
    end

    subgraph "Execution Layer"
        B --> P[Portfolio Optimizer]
        V --> P
        BE --> P
        P --> |Guardrails| G[Sector Caps + Stop Loss]
        G --> EX[Final Portfolio Selection]
    end

    style AF fill:#f9f,stroke:#333,stroke-width:4px
    style S2 fill:#bbf,stroke:#333,stroke-width:2px
    style RG fill:#bfb,stroke:#333,stroke-width:2px
```

### Key Components:
- **Technical Expert**: Provides the baseline trend-following momentum.
- **Foundation Expert**: Leverages the power of pre-trained time-series transformers (Chronos-T5) fine-tuned via Low-Rank Adaptation (LoRA) to capture non-linear alpha.
- **Risk Expert**: Uses a Hidden Markov Model (HMM) to detect market regimes and dynamically adjust the trust (weights) assigned to the Technical vs. Foundation experts.
