# RAMT Core Audit

Read-only audit of the repository state. Every behavioral claim below cites **file path and line number** in the form `(path:Lx–Ly)` or `(path:Lx)`.

---

## Section 1 — Data Pipeline

### 1.1 Processed Parquet files (`data/processed/*.parquet`)

**Empirical scan (current workspace):** 201 `*.parquet` files under `data/processed/`. For **200** equity feature files (example: `360ONE_NS_features.parquet`), the column **names and dtypes** are:

| Column | dtype (pyarrow) |
|--------|-----------------|
| Date | timestamp[ms] |
| Ticker | large_string |
| Open, High, Low, Close, Adj Close | double |
| Volume | int64 |
| Ret_1d, Ret_5d, Ret_21d, Realized_Vol_20, RSI_14, BB_Dist, Volume_Surge | double |
| Macro_INDIAVIX_Ret1d_L1, Macro_CRUDE_Ret1d_L1, Macro_USDINR_Ret1d_L1, Macro_SP500_Ret1d_L1 | double |
| Monthly_Alpha | double |
| Daily_Return | double |
| HMM_Regime | double |
| Sector | large_string |
| Sector_Alpha | double |

**One file differs only in column order (same columns):** `_NSEI_features.parquet` has `Monthly_Alpha`, `Sector_Alpha`, `Daily_Return`, `HMM_Regime`, `Sector` in a different order than the equity files (see `features/feature_engineering.py` canonical order `FEATURE_OUTPUT_COLUMNS` at `(features/feature_engineering.py:L186–L208)`).

**Row count and date range (scan):** All 201 files report **1533 rows** each; Date range **2020-01-01 through 2026-03-11** (inclusive on daily timestamps). This aligns with feature pipeline `START_DATE` / `END_DATE_EXCLUSIVE` in `(features/feature_engineering.py:L38–L40)`.

**Classification (derived from `dataset.py` + `feature_engineering.py`):**

- **Model input features (scaled in training):** `ALL_FEATURE_COLS` in `(models/ramt/dataset.py:L18–L35)` — 11 columns: `Ret_1d`, `Ret_5d`, `Ret_21d`, `RSI_14`, `BB_Dist`, `Volume_Surge`, and the four `Macro_*_Ret1d_L1` columns.
- **Targets / labels:** `TARGET_COL` = `Sector_Alpha` with fallback to `Monthly_Alpha` if `Sector_Alpha` missing `(models/ramt/dataset.py:L40–L42, L231–L237, L351–L362)`; auxiliary `Daily_Return` for the daily head `(models/ramt/dataset.py:L240–L242, L365–L372)`.
- **Metadata / other inputs:** `Date`, `Ticker`, raw OHLCV columns; `HMM_Regime` passed to the model **separately** from `X` `(models/ramt/dataset.py:L37–L38, L15–L16)`; `Sector` used for sector-neutral scaling routing `(models/ramt/train_ranking.py:L207–L223, L537–L567)`.

**Note on “36 features”:** The current codebase defines **11** parquet columns as the model’s feature vector (`ALL_FEATURE_COLS`). The full engineered **output schema** lists **24** columns in `FEATURE_OUTPUT_COLUMNS` `(features/feature_engineering.py:L186–L208)`. A count of **36** engineered features is **NOT FOUND IN CURRENT CODEBASE**.

---

### 1.2 How `data/download.py` builds the raw dataset

- **Source:** Yahoo Finance via `yfinance` `(data/download.py:L216–L231)`.
- **Universe:** Not a hardcoded list in the running loop — `TICKERS = load_tickers_from_file(NIFTY_200_TICKERS_FILE)` where `NIFTY_200_TICKERS_FILE` is `data/nifty200_tickers.txt` `(data/download.py:L90–L131)`.
- **Date range:** `START_DATE = "2015-01-01"`, `END_DATE = "2026-01-01"` (yfinance `end` is exclusive) `(data/download.py:L127–L128, L198–L209, L472–L474)`.
- **Per-ticker output:** `download_one_ticker` adds `Log_Return` and `Ticker`, saves CSV under `data/raw/` `(data/download.py:L254–L266, L285–L286, L490–L491)`.
- **Macro tickers:** Separate downloads for `MACRO_TICKERS` `(data/download.py:L119–L125, L507–L541)`.

The **processed Parquet pipeline** used by `features/feature_engineering.py` is **not** driven by `data/download.py`; it expects raw Parquet inputs from `scripts/fetch_nifty200.py` (see module docstring `(features/feature_engineering.py:L5–L11, L55–L56)`).

---

### 1.3 How `features/feature_engineering.py` builds features

Below: **each engineered column** (excluding `Date`, `Ticker`, raw OHLCV), with **formula**, **sources**, and **lookback** where applicable. All use **Adj Close** where noted in source.

| Feature | Formula / rule (code) | Source columns | Lookback |
|--------|------------------------|-----------------|----------|
| `Ret_1d` | `ln(AdjClose_t / AdjClose_{t-1})` with non-positive ratios masked out | `Adj Close` | 1 day |
| `Ret_5d` | `ln(AdjClose_t / AdjClose_{t-5})` | `Adj Close` | 5 days |
| `Ret_21d` | `ln(AdjClose_t / AdjClose_{t-21})` | `Adj Close` | 21 days |
| `Realized_Vol_20` | Rolling std of `Ret_1d` with window 20 | `Ret_1d` | 20 rows |
| `RSI_14` | Wilder-style EWM with `alpha=1/14` on gains/losses from `close.diff()` | `Adj Close` | EWM (implicit 14) |
| `BB_Dist` | `(AdjClose - MA20) / (2 * STD20)`; `std=0` denom → NaN | `Adj Close` | 20 rows |
| `Volume_Surge` | `Volume / SMA20(Volume)`; SMA 0 → NaN | `Volume` | 20 rows |
| `Macro_*_Ret1d_L1` | For each macro: `ln(AdjClose_t/AdjClose_{t-1})` then **shift(1)** (one trading day lag), left-joined, ffill, NaN→0 | Macro `Adj Close` | 1-day lag |
| `Monthly_Alpha` | `ln(P_{t+21}/P_t) - ln(N_{t+21}/N_t)` for stock and NIFTY | `Adj Close` (stock + benchmark) | 21 days forward |
| `Daily_Return` | `Ret_1d.shift(-1)` | `Ret_1d` | 1 day forward |
| `HMM_Regime` | See §1.5 | `Ret_1d`, `Realized_Vol_20` | expanding |
| `Sector` | `get_sector(ticker)` | ticker string | — |
| `Sector_Alpha` | `Monthly_Alpha - median(Monthly_Alpha \| Date, Sector)` | `Monthly_Alpha`, `Sector` | panel |

Code references:

- Returns: `(features/feature_engineering.py:L374–L383)`
- Realized vol: `(features/feature_engineering.py:L386–L389)`
- RSI: `(features/feature_engineering.py:L528–L541)`
- Bollinger distance: `(features/feature_engineering.py:L544–L554)`
- Volume surge: `(features/feature_engineering.py:L557–L562)`
- Macro merge: `(features/feature_engineering.py:L565–L584)`
- Monthly alpha: `(features/feature_engineering.py:L587–L605)`
- Daily return target: `(features/feature_engineering.py:L392–L398)`
- HMM: `(features/feature_engineering.py:L462–L525)`
- Sector column: `(features/feature_engineering.py:L230–L232)`
- Sector alpha panel: `(features/feature_engineering.py:L608–L645)`

Warm-up NaNs for several columns are filled with **0.0** before column selection `(features/feature_engineering.py:L234–L248)`.

---

### 1.4 `SECTOR_MAP` — full dict

Defined in `(features/sectors.py:L15–L125)`:

```python
SECTOR_MAP: dict[str, str] = {
    # --- Banking & Financials ---
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "KOTAKBANK": "BANK",
    "AXISBANK": "BANK", "SBIN": "BANK", "INDUSINDBK": "BANK",
    "BANKBARODA": "BANK", "CANBK": "BANK", "PNB": "BANK", "BANKINDIA": "BANK",
    "FEDERALBNK": "BANK", "IDFCFIRSTB": "BANK", "AUBANK": "BANK",
    "IDBI": "BANK", "RBLBANK": "BANK", "BANDHANBNK": "BANK", "YESBANK": "BANK",
    "IOB": "BANK", "UNIONBANK": "BANK", "CENTRALBK": "BANK", "INDIANB": "BANK",

    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "BAJAJHLDNG": "NBFC",
    "CHOLAFIN": "NBFC", "SHRIRAMFIN": "NBFC", "MUTHOOTFIN": "NBFC",
    "PFC": "NBFC", "RECLTD": "NBFC", "LICHSGFIN": "NBFC", "MFSL": "NBFC",
    "360ONE": "NBFC", "ABCAPITAL": "NBFC", "IRFC": "NBFC", "POONAWALLA": "NBFC",
    "HDFCAMC": "NBFC", "CDSL": "NBFC", "BSE": "NBFC", "MCX": "NBFC",

    "SBILIFE": "INSURANCE", "HDFCLIFE": "INSURANCE", "ICICIGI": "INSURANCE",
    "ICICIPRULI": "INSURANCE", "LICI": "INSURANCE", "SBICARD": "INSURANCE",

    # --- Information Technology ---
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "LTIM": "IT", "LTTS": "IT", "PERSISTENT": "IT",
    "COFORGE": "IT", "MPHASIS": "IT", "OFSS": "IT", "KPITTECH": "IT",
    "TATAELXSI": "IT", "TATATECH": "IT",

    # --- Energy (Oil & Gas) ---
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "IOC": "ENERGY", "BPCL": "ENERGY",
    "HINDPETRO": "ENERGY", "GAIL": "ENERGY", "OIL": "ENERGY", "PETRONET": "ENERGY",
    "MGL": "ENERGY", "IGL": "ENERGY", "GUJGASLTD": "ENERGY",

    # --- Power & Utilities ---
    "POWERGRID": "POWER", "NTPC": "POWER", "TATAPOWER": "POWER",
    "ADANIPOWER": "POWER", "ADANIGREEN": "POWER", "ADANIENSOL": "POWER",
    "COALINDIA": "POWER", "NHPC": "POWER", "SJVN": "POWER", "JSWENERGY": "POWER",
    "TORNTPOWER": "POWER", "CESC": "POWER", "NTPCGREEN": "POWER",

    # --- FMCG (Consumer Staples) ---
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "TATACONSUM": "FMCG",
    "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG", "VBL": "FMCG",
    "UBL": "FMCG", "PATANJALI": "FMCG", "EMAMILTD": "FMCG", "RADICO": "FMCG",
    "UNITDSPR": "FMCG",

    # --- Auto ---
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO", "EICHERMOT": "AUTO",
    "HEROMOTOCO": "AUTO", "BAJAJ_AUTO": "AUTO", "ASHOKLEY": "AUTO",
    "BHARATFORG": "AUTO", "TVSMOTOR": "AUTO", "MOTHERSON": "AUTO",
    "EXIDEIND": "AUTO", "MRF": "AUTO", "BALKRISIND": "AUTO", "APOLLOTYRE": "AUTO",
    "TIINDIA": "AUTO", "BOSCHLTD": "AUTO", "UNOMINDA": "AUTO", "SONACOMS": "AUTO",

    # --- Pharma & Healthcare ---
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "LUPIN": "PHARMA", "AUROPHARMA": "PHARMA",
    "TORNTPHARM": "PHARMA", "ALKEM": "PHARMA", "ZYDUSLIFE": "PHARMA",
    "GLENMARK": "PHARMA", "BIOCON": "PHARMA", "MANKIND": "PHARMA",
    "ABBOTINDIA": "PHARMA", "GLAND": "PHARMA", "IPCALAB": "PHARMA",
    "LAURUSLABS": "PHARMA", "SYNGENE": "PHARMA", "NATCOPHARM": "PHARMA",

    "APOLLOHOSP": "HEALTH", "MAXHEALTH": "HEALTH", "FORTIS": "HEALTH",
    "LALPATHLAB": "HEALTH", "METROPOLIS": "HEALTH",

    # --- Metals & Mining ---
    "TATASTEEL": "METAL", "HINDALCO": "METAL", "JSWSTEEL": "METAL",
    "VEDL": "METAL", "SAIL": "METAL", "NMDC": "METAL", "HINDZINC": "METAL",
    "JINDALSTEL": "METAL", "APLAPOLLO": "METAL", "JSL": "METAL",
    "NATIONALUM": "METAL",

    # --- Telecom & Media ---
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM", "INDUSTOWER": "TELECOM",
    "TATACOMM": "TELECOM", "ZEEL": "MEDIA", "SUNTV": "MEDIA",

    # --- Cement & Construction Materials ---
    "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT", "SHREECEM": "CEMENT",
    "AMBUJACEM": "CEMENT", "ACC": "CEMENT", "JKCEMENT": "CEMENT",
    "DALBHARAT": "CEMENT", "RAMCOCEM": "CEMENT",

    # --- Infrastructure / Construction / Capital Goods ---
    "LT": "INFRA", "ADANIPORTS": "INFRA", "GMRAIRPORT": "INFRA",
    "IRCTC": "INFRA", "CONCOR": "INFRA", "RVNL": "INFRA", "IRCON": "INFRA",
    "KEC": "INFRA", "NBCC": "INFRA", "GMRINFRA": "INFRA",

    "SIEMENS": "CAPGOODS", "ABB": "CAPGOODS", "BEL": "CAPGOODS",
    "BHEL": "CAPGOODS", "HAL": "CAPGOODS", "BDL": "CAPGOODS",
    "CGPOWER": "CAPGOODS", "POLYCAB": "CAPGOODS", "HAVELLS": "CAPGOODS",
    "CROMPTON": "CAPGOODS", "SUPREMEIND": "CAPGOODS", "VOLTAS": "CAPGOODS",
    "BLUESTARCO": "CAPGOODS", "CUMMINSIND": "CAPGOODS", "THERMAX": "CAPGOODS",
    "SOLARINDS": "CAPGOODS", "HONAUT": "CAPGOODS", "AIAENG": "CAPGOODS",
    "COCHINSHIP": "CAPGOODS", "MAZDOCK": "CAPGOODS", "GRSE": "CAPGOODS",

    # --- Consumer Discretionary / Retail ---
    "TITAN": "DISCRETIONARY", "DMART": "DISCRETIONARY", "TRENT": "DISCRETIONARY",
    "JUBLFOOD": "DISCRETIONARY", "ZOMATO": "DISCRETIONARY", "PAYTM": "DISCRETIONARY",
    "NYKAA": "DISCRETIONARY", "VMART": "DISCRETIONARY", "ABFRL": "DISCRETIONARY",
    "PAGEIND": "DISCRETIONARY", "RELAXO": "DISCRETIONARY", "BATAINDIA": "DISCRETIONARY",
    "INDHOTEL": "DISCRETIONARY", "DIXON": "DISCRETIONARY", "HAVISHA": "DISCRETIONARY",
    "PGEL": "DISCRETIONARY", "KALYANKJIL": "DISCRETIONARY",

    # --- Chemicals / Specialty / Paints ---
    "PIDILITIND": "CHEM", "ASIANPAINT": "CHEM", "BERGEPAINT": "CHEM",
    "UPL": "CHEM", "SRF": "CHEM", "PIIND": "CHEM", "DEEPAKNTR": "CHEM",
    "ATUL": "CHEM", "AARTIIND": "CHEM", "NAVINFLUOR": "CHEM",
    "SOLARA": "CHEM", "GODREJIND": "CHEM", "LINDEINDIA": "CHEM",

    # --- Real Estate ---
    "DLF": "REALTY", "GODREJPROP": "REALTY", "OBEROIRLTY": "REALTY",
    "PRESTIGE": "REALTY", "PHOENIXLTD": "REALTY", "LODHA": "REALTY",
    "BRIGADE": "REALTY",

    # --- Textiles / Paper / Misc ---
    "ASTRAL": "BUILDING", "PRINCEPIPE": "BUILDING", "CERA": "BUILDING",
    "KAJARIACER": "BUILDING", "SOMANYCERA": "BUILDING",
}
```

`DEFAULT_SECTOR = "OTHER"` `(features/sectors.py:L127)`. Lookup strips suffixes and uppercases `(features/sectors.py:L138–L145)`.

---

### 1.5 `Sector_Alpha` computation

After all per-ticker parquets are written, `apply_sector_alpha_panel` builds a panel of all tickers, computes **groupby** `["Date", "Sector"]` **median** of `Monthly_Alpha`, then:

`Sector_Alpha = Monthly_Alpha - med` with `med = panel.groupby(["Date", "Sector"], sort=False)["Monthly_Alpha"].transform("median")` `(features/feature_engineering.py:L622–L633)`.

---

### 1.6 `HMM_Regime` computation

- **Model type:** `GaussianHMM` from `hmmlearn` `(features/feature_engineering.py:L24, L434–L442)`.
- **`n_states`:** `n_components=3` `(features/feature_engineering.py:L434–L435)`.
- **Observations:** `X = [Ret_1d, Realized_Vol_20]` standardized **on the expanding history up to each end position** (mean/std of the prefix), then HMM fit and predict on that prefix `(features/feature_engineering.py:L488–L498)`.
- **Training window:** Expanding from index `HMM_MIN_OBS - 1` onward; `HMM_MIN_OBS = 60` `(features/feature_engineering.py:L53, L476–L488)`.
- **Causality:** For each date index `end_pos`, only data `X_hist` up to `end_pos` is used; the assigned regime is stored at `out.loc[idx[end_pos], "HMM_Regime"]` `(features/feature_engineering.py:L488–L516)`.
- **Semantic mapping:** Raw states mapped to `{0,1,2}` via `_semantic_hmm_mapping` (bull=1 highest mean log-return, bear=2 lowest, high-vol/mid=0) `(features/feature_engineering.py:L401–L426, L500–L507)`.

---

## Section 2 — Scaling & Normalization

### 2.1 `SectorNeutralScaler`

Defined in `(models/ramt/train_ranking.py:L479–L535)`.

- **Fit:** Fits a **global** `RobustScaler` on all rows `X`, then for each unique sector string with **at least** `min_samples_per_sector` (default **500**) rows, fits a **per-sector** `RobustScaler` `(models/ramt/train_ranking.py:L502–L516)`.
- **Transform:** Uses `per_sector[active_sector]` if present; otherwise `global_fallback` `(models/ramt/train_ranking.py:L523–L530)`. Active sector is set via `set_active_sector` before transforming windows in `LazyMultiTickerSequenceDataset.__getitem__` `(models/ramt/dataset.py:L423–L427)`.

### 2.2 Which features get scaled

- **Scaled:** Only `ALL_FEATURE_COLS` (11 features) `(models/ramt/dataset.py:L18–L35, L247–L248, L375–L376)`.
- **Not scaled in `X`:** `HMM_Regime` is **not** in `X`; it is passed separately `(models/ramt/dataset.py:L15–L16, L37–L38)`.
- **Target scaling:** `y_scaler` is a `RobustScaler` fit on **monthly** training labels (after winsorization bounds applied) `(models/ramt/train_ranking.py:L50–L63, L236–L268)`.

### 2.3 Save / load paths

- Written by `save_ramt_inference_artifacts`: `results/ramt_scaler.joblib`, `results/ramt_y_scaler.joblib` `(models/ramt/train_ranking.py:L1058–L1062)`.
- Fold snapshots: `ramt_scaler_{tag}.joblib`, `ramt_y_scaler_{tag}.joblib` `(models/ramt/train_ranking.py:L1064–L1068)`.

### 2.4 Training-only statistics (leakage)

- **Feature scaler:** Fit on `train_keys_final` only (`_fit_sector_neutral_scaler_on_train` or `_fit_scaler_on_train`) `(models/ramt/train_ranking.py:L231–L234, L456–L476, L537–L567)`.
- **Y winsor bounds:** `lo_b`, `hi_b` from **1st/99th percentile of training monthly labels only** `(models/ramt/train_ranking.py:L236–L241)`.
- **`y_scaler`:** Fit on training keys after winsorization applied to stored raw arrays `(models/ramt/train_ranking.py:L258–L268, L50–L63)`.
- **Validation split:** `val_keys` is last 6 months of the training window; excluded from `train_keys_final` `(models/ramt/train_ranking.py:L202–L205)`.

Single-ticker `RAMTDataset.get_fold_loaders` also fits `RobustScaler` on **train indices only** `(models/ramt/dataset.py:L252–L272)`.

---

## Section 3 — Dataset Construction (`models/ramt/dataset.py`)

- **`seq_len`:** Default **30** `(models/ramt/dataset.py:L168–L173, L206–L207, L300)`.
- **Batch construction:** `LazyMultiTickerSequenceDataset` samples are keyed by `(ticker, row_index)` — each sequence is **one ticker**, **contiguous dates** of length `seq_len` ending at `i`; **not** a cross-sectional “same date, many tickers” tensor. Cross-sectional ranking happens **inside the training loop** by grouping on the date id `db` `(models/ramt/dataset.py:L388–L456, models/ramt/train_ranking.py:L817–L823)`.
- **Walk-forward (single-ticker helper):** `get_walk_forward_indices(init_train_frac=0.6, step_size=63)` — initial train `[0, train_end)`, test `[train_end, train_end+step_size)`, advancing `train_end` by `step_size` `(models/ramt/dataset.py:L305–L317)`.
- **Walk-forward (multi-ticker production):** Implemented in `combined_walk_forward` in `train_ranking.py` (see §6) — uses `training_step` (default 126) for segment boundaries and `rebalance_step` (default 21) for inference grid `(models/ramt/train_ranking.py:L1077–L1099, L1114–L1172)`.
- **Regime to model:** Passed as separate tensor `r` / `rb`, not concatenated into `X` `(models/ramt/dataset.py:L447–L455, models/ramt/model.py:L136–L161)`.
- **Target variable:** `TARGET_COL = "Sector_Alpha"` with fallback to `Monthly_Alpha` `(models/ramt/dataset.py:L40–L42, L231–L249)`.
- **NaN handling:** Rows dropped if `eff` target missing; if `Daily_Return` present, rows with missing `Daily_Return` dropped `(models/ramt/dataset.py:L239–L242, L372)`. `clip_target` exists but is **not** called in `LazyTickerStore` loading path `(models/ramt/dataset.py:L148–L156, L220–L249)`.

---

## Section 4 — Model Architecture

### 4.1 Input shape at entry

`x`: `(batch, seq_len, len(ALL_FEATURE_COLS))` with `len(ALL_FEATURE_COLS) == 11` `(models/ramt/model.py:L30–L31, models/ramt/encoder.py:L111–L116)`.

### 4.2 Encoders (`models/ramt/encoder.py`)

| Component | Input dim | Output dim | Activation / structure |
|-----------|-----------|-------------|----------------------|
| `FeedForwardEncoder` (price) | `len(PRICE_COLS)=3` | `group_dim` (default 32) | Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm `(models/ramt/encoder.py:L18–L30)` |
| `FeedForwardEncoder` (tech) | 2 | 32 | same |
| `FeedForwardEncoder` (volume) | 1 | 32 | same |
| `FeedForwardEncoder` (macro) | 4 | 32 | same |
| `RegimeEncoder` | 3 classes | 32 | Embedding + LayerNorm `(models/ramt/encoder.py:L36–L48)` |
| `TickerEncoder` | `max(1, len(TICKER_LIST))` | 32 | Embedding + LayerNorm `(models/ramt/encoder.py:L51–L61)` |

**Extra input scaling:** `Ret_21d` channel multiplied by `RET_21D_INPUT_SCALE = 1.65` after slicing `(models/ramt/encoder.py:L13–L15, L117–L119)`.

### 4.3 Fusion

Concatenation of **six** `group_dim` embeddings (price, tech, volume, macro, regime, ticker or zeros) → linear fusion to `embed_dim` (default 64) `(models/ramt/encoder.py:L92–L98, L132–L142)`.

### 4.4 `RegimeCrossAttention` (`models/ramt/moe.py:L148–L206`)

- **Query:** `regime_query` embedding indexed by regime id, expanded to `(B, T, D)` `(models/ramt/moe.py:L179–L203)`.
- **Key / Value:** `features` (same tensor for both) `(models/ramt/moe.py:L203)`.
- **Heads / dim:** `num_heads` default **4**, `embed_dim` default **64** (`MultiheadAttention` defaults to `embed_dim // num_heads` for head dim) `(models/ramt/moe.py:L171–L186)`.
- **Masking:** No `attn_mask` or `key_padding_mask` passed to `cross_attn` — **no masking** `(models/ramt/moe.py:L181–L203)`.

### 4.5 Transformer experts (`models/ramt/moe.py:L74–L127`)

Each `ExpertTransformer`: `TransformerEncoder` with `num_layers=2`, `nhead=4`, `dim_feedforward=128`, `dropout`, `batch_first=True`, `norm_first=True` `(models/ramt/moe.py:L93–L114)`. Output is **last timestep** `(models/ramt/moe.py:L121–L124)`.

### 4.6 MoE (`models/ramt/moe.py:L295–L409`)

- **`n_experts`:** 3 `(models/ramt/moe.py:L328–L357)`.
- **`top_k`:** **NOT FOUND IN CURRENT CODEBASE** — gating produces **dense** softmax weights over all 3 experts `(models/ramt/moe.py:L287–L407)`.
- **Routing:** `GatingNetwork` concatenates `context` (last timestep of sequence) with **one-hot regime**, MLP → **softmax** over experts `(models/ramt/moe.py:L232–L292, L369–L407)`.
- **Load balancing:** No auxiliary load-balancing loss in MoE forward **NOT FOUND IN CURRENT CODEBASE**.

### 4.7 Output heads (`models/ramt/model.py:L119–L134, L165–L168`)

- **`monthly_head`:** LayerNorm → Linear(64→32) → ReLU → Dropout → Linear(32→1) — **no explicit final activation** (identity).
- **`daily_head`:** Same structure.
- **Final outputs:** `pred_monthly`: `(batch, 1)`, `pred_daily`: `(batch, 1)`, `gate_weights`: `(batch, num_experts)` `(models/ramt/model.py:L165–L168)`.

---

## Section 5 — Loss Function (`models/ramt/losses.py` + training usage)

### 5.1 `TournamentRankingLoss`

Exact forward `(models/ramt/losses.py:L99–L121)`:

- `y_diff = y.unsqueeze(1) - y.unsqueeze(0)`, `p_diff` analogous.
- **Pair sampling:** All ordered pairs with `y_diff > 0` (upper triangle of true ordering), **not** random subsampling.
- **Margin:** `hinge = relu(margin - p_diff)` with default `margin=0.02` `(models/ramt/losses.py:L99–L117)`.
- **Magnitude weight:** `weight = abs(y_diff) * mask` (only where `y_diff > 0`).
- **Reduction:** `(hinge * weight).sum() / (weight.sum() + 1e-8)` `(models/ramt/losses.py:L119–L121)`.

### 5.2 Auxiliary loss terms (training loop)

In `_train_one_epoch` with `USE_TOURNAMENT_LOSS = True`, total loss is:

`loss = rank_loss + AUX_DAILY_WEIGHT * mse_d` where `mse_d` is **weighted MSE** on **daily head** vs `yb_d`, with regime-0 weight `HIGH_VOL_SAMPLE_WEIGHT` and `_time_decay_weights` on dates `(models/ramt/train_ranking.py:L121–L125, L642–L653, L813–L839)`.

`CombinedLoss` (MSE + directional) is **instantiated** `(models/ramt/train_ranking.py:L283–L284)` but **`criterion` is never invoked inside `_train_one_epoch` or `_eval_loss`** — those functions recompute rank + daily MSE only. So **CombinedLoss is not used in the active training/eval path** `(models/ramt/train_ranking.py:L779–L862, L865–L927)`.

### 5.3 Scaled vs unscaled at loss time

- **Ranking:** Predictions are **inverse-transformed** to unscaled monthly alpha via `_monthly_pred_unscaled` before `TournamentRankingLoss` when `USE_TOURNAMENT_LOSS` `(models/ramt/train_ranking.py:L719–L748, L746–L748)`.
- **True labels for ranking:** `yb_m_raw` (winsorized raw) `(models/ramt/train_ranking.py:L794–L805, L447–L455)`.

---

## Section 6 — Training Loop (`models/ramt/train_ranking.py`)

**Note:** There is **no** `models/ramt/train.py` in the repo — only `train_ranking.py` (glob search). The entry `if __name__ == "__main__"` runs `combined_walk_forward()` `(models/ramt/train_ranking.py:L1416–L1422)`.

| Topic | Value / behavior | Source |
|-------|------------------|--------|
| Optimizer | `AdamW`, `lr=WARMUP_LR_END` (1e-4), `weight_decay=WEIGHT_DECAY` (1e-4) | `(models/ramt/train_ranking.py:L107, L284)` |
| Betas | **Not set in code** — PyTorch `AdamW` defaults apply | `(models/ramt/train_ranking.py:L284)` — no `betas=` kwarg |
| LR scheduler | `ReduceLROnPlateau`, `mode=min`, `factor=0.5`, `patience=2`, `min_lr=1e-7`; stepped on **val loss** after warmup | `(models/ramt/train_ranking.py:L285–L314)` |
| Warmup | Linear LR from `WARMUP_LR_START` (1e-7) to `WARMUP_LR_END` (1e-4) over `WARMUP_STEPS` (500) **optimizer steps** | `(models/ramt/train_ranking.py:L102–L106, L294–L300, L795–L800)` |
| Gradient clipping | Yes, `max_norm=GRAD_CLIP` (1.0) | `(models/ramt/train_ranking.py:L110, L842)` |
| Mixed precision | **NOT FOUND IN CURRENT CODEBASE** (no `torch.cuda.amp` / autocast in training) | grep `autocast` in `*.py` |
| Epochs | `MAX_EPOCHS = 30`, overridable by `max_epochs` argument | `(models/ramt/train_ranking.py:L108, L294–L295)` |
| Early stopping | On **validation loss** `v`; `PATIENCE = 8`; `best` improved when `v < best` (no `min_delta`) | `(models/ramt/train_ranking.py:L109, L305–L332)` |
| Checkpointing | Best **val loss** state dict kept in memory; `save_ramt_inference_artifacts` writes `results/ramt_model_state.pt` when `plot_dir` set and **seg_idx == 0** only inside `combined_walk_forward` — fold-tagged copies when `artifact_dir` set | `(models/ramt/train_ranking.py:L325–L346, L1136–L1158, L1020–L1074)` |
| Validation metric | **Combined val loss** (same formula as train: rank + `AUX_DAILY_WEIGHT * mse_d`), **not** Sharpe/DA | `(models/ramt/train_ranking.py:L865–L927)` |
| Device | `mps` if available, else `cuda`, else `cpu` | `(models/ramt/train_ranking.py:L93–L98)` |

---

## Section 7 — Backtest Logic

**File location:** There is **no** `backtest.py` at the repository root. Portfolio code lives at **`models/backtest.py`** (this is what `models/run_final_2024_2026.py` imports `(models/run_final_2024_2026.py:L33)`).

### 7.1 Walk-forward windows (training vs backtest module)

`models/backtest.py` **does not** implement ML walk-forward training. That logic is in **`combined_walk_forward`** `(models/ramt/train_ranking.py:L1077–L1209)`:

- **Outer test cadence:** `segment_starts = _rebalance_dates_21d(nifty_path, TEST_START, TEST_END, step_size=int(training_step))` with default **`training_step=126`** trading days between segment starts `(models/ramt/train_ranking.py:L1114–L1117)`.
- **Training window:** Each segment trains on samples with dates in **`[TRAIN_START, train_end_inclusive]`** where `TRAIN_START = "2015-01-01"` and `train_end_inclusive` is `TRAIN_END` for the first segment, then the last trading day before the next segment start `(models/ramt/train_ranking.py:L85–L86, L1126–L1132, L1136–L1144)`.
- **Inference grid inside a segment:** `pred_dates` uses `_rebalance_dates_21d(..., step_size=int(rebalance_step))` default **`rebalance_step=21`** `(models/ramt/train_ranking.py:L1167–L1172)`.

The naming **`train_days` / `test_days` / `stride`** as separate integer knobs is **NOT FOUND IN CURRENT CODEBASE** (replaced by `training_step`, `rebalance_step`, and calendar-derived segments).

### 7.2 `run_backtest_daily` (primary daily backtest)

- **Rebalance grid:** NIFTY trading calendar `cal[::step_size]` default `step_size=21` `(models/backtest.py:L487–L555)` — note: file path is `models/backtest.py` (not repo root `backtest.py`).
- **Top-N:** `nlargest(n_sel, "predicted_alpha")` with `n_sel` from regime (3 for high vol, 5 cap in bear, else `top_n`) `(models/backtest.py:L598–L635)`.
- **Position sizing:** `position_size` 0.2 / 0.5 / 1.0 by regime; optional Kelly scaling `kelly_scale_position` `(models/backtest.py:L598–L668)`. Weights: softmax over `predicted_alpha` if `use_kelly_weights`, else equal weight per name capped by `max_weight` `(models/backtest.py:L669–L682)`.
- **Friction:** `friction_fee = pv_start * invested * effective_friction_rate`; `rebalance_friction_rate` default `REAL_2026_REBALANCE_FRICTION_RATE = 0.0022`; if passed value equals `LEGACY_REBALANCE_FRICTION_RATE` (0.002), replaced with 0.0022 `(models/backtest.py:L29–L31, L542–L544, L693–L695)`.
- **Compounding:** `pv = pv_start * (1.0 + port_ret)` each window `(models/backtest.py:L516–L518, L723)`.
- **HMM regime:** Loaded from `_NSEI_features.parquet` `HMM_Regime` aligned to rebalance dates `(models/backtest.py:L227–L245, L548–L559)`. Used for **position sizing and top-N**, not separate metrics in this function.
- **Sharpe / MaxDD / DA:** `run_backtest_daily` **does not** compute Sharpe or max drawdown at the end — it returns per-window rows `(models/backtest.py:L748–L752)`. **Monthly** `run_backtest` computes Sharpe, max DD, win rate from **monthly** returns `(models/backtest.py:L462–L469)`.

**Directional accuracy (DA):** **NOT FOUND IN CURRENT CODEBASE** in `models/backtest.py`.

---

## Section 8 — Known Flaws / Open Questions

### 8.1 TODO / FIXME

- Grep for `TODO`, `FIXME`, `XXX`, `HACK` across `*.py`, `*.md`, `*.ipynb`, `*.txt`: **no matches** in the current codebase.

### 8.2 Hardcoded magic numbers (should be config)

Examples (non-exhaustive): `RET_21D_INPUT_SCALE = 1.65` `(models/ramt/encoder.py:L13–L15)`; `LEARNING_RATE`, `WARMUP_*`, `RANKING_MARGIN_ALPHA`, `AUX_DAILY_WEIGHT`, `MIN_CROSSSECTION_SIZE`, `HIGH_VOL_SAMPLE_WEIGHT` `(models/ramt/train_ranking.py:L100–L125)`; friction rates `(models/backtest.py:L29–L31)`; `HMM_MIN_OBS = 60` `(features/feature_engineering.py:L53)`.

### 8.3 References to `.pt` artifacts

- Canonical inference path: `results/ramt_model_state.pt` `(models/permutation_importance.py:L114)`, `(models/inspect_attention.py:L102)`, `(models/attention_consistency_report.py:L49)`, `(dashboard/app.py:L1369)`.
- Walk-forward also writes `ramt_model_state_{tag}.pt` e.g. `wf_seg_01` `(models/ramt/train_ranking.py:L1064–L1067, L1157)`.

### 8.4 Inconsistencies (docs / comments vs code)

- **`RAMTModel.forward` docstring** claims return is “next-day return prediction” for `(batch,1)` — but the first return is **`pred_monthly`** from `monthly_head`, and the second is **`pred_daily`** `(models/ramt/model.py:L136–L168)`.
- **`combined_walk_forward(start, end)`** parameters **`start` and `end` are unused** in the function body; training uses module-level `TRAIN_START` / `TEST_*` constants instead `(models/ramt/train_ranking.py:L85–L88, L1077–L1140)`.
- **`RAMTDataset` class docstring** says “Primary target: Monthly_Alpha” but `TARGET_COL` is **`Sector_Alpha`** `(models/ramt/dataset.py:L194–L199, L40–L42)`.
- **`data/download.py` module docstring** mentions JPM / EPIGRAL / mixed tickers; actual `TICKERS` load from **`data/nifty200_tickers.txt`** `(data/download.py:L1–L6, L90–L131)`.
- **`moe.py` `__main__` test assertions** expect `ExpertTransformer` output shape `(batch_size, 1)` but `forward` returns `(batch, embed_dim)` `(models/ramt/moe.py:L457–L462)` — the test block is inconsistent with the implementation.
- **`train_fixed_and_predict`** raises `NotImplementedError` immediately; dead code follows after the raise `(models/ramt/train_ranking.py:L1227–L1231)`.

---

*End of audit document.*
