# RAMT — concepts & notes (living document)

This file collects explanations for ideas, code, and methods used in this repo.  
Ask in chat to add a new topic; it gets appended here in plain language.

---

## Why we use encoders (`MultimodalEncoder`)

Encoders exist so downstream parts of the model (fusion, attention, prediction head) receive **consistent, learned representations** instead of raw feature columns in one flat vector.

1. **One space for many input types**  
   The 27 features are grouped (returns, volatility, technicals, momentum, volume, regime, cross-asset). Scales and meaning differ. Each group encoder maps its slice to the **same width** (`group_dim`, e.g. 32) so we can **concatenate** and **fuse** with one linear layer into `embed_dim` for the transformer.

2. **Nonlinear preprocessing per group**  
   A small feedforward stack lets the model learn **group-specific** transforms (e.g. how volatility features interact within the vol group) before mixing groups. Global nonlinearity-only-later can be harder to train on heterogeneous inputs.

3. **Regime is categorical**  
   `HMM_Regime` is **categorical** (states 0, 1, 2), not a continuous magnitude. **`nn.Embedding`** gives each state its own learned vector instead of treating “2” as twice “1”. Continuous groups use linear layers on real values; regime uses embedding after integer labels are recovered from the scaled column.

4. **Fixed width for the transformer**  
   Attention blocks expect a fixed **token size** (`embed_dim`). Encoders adapt `(batch, seq, 27)` → `(batch, seq, embed_dim)`.

**Short summary:** specialize per modality, align dimensions, treat regime as categories, then fuse.

---

<!-- New sections go below this line -->
