"""
Hand-curated NIFTY 200 -> sector mapping, sourced from NSE sector index
constituents (public info on nseindia.com).

Used by:
  - features/feature_engineering.py — attach Sector column to each parquet
  - models/ramt/train_ranking.py — SectorNeutralScaler fits per-sector stats

Any ticker not in SECTOR_MAP falls into DEFAULT_SECTOR so training still runs
end-to-end; add entries here as the universe evolves.
"""

from __future__ import annotations

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

DEFAULT_SECTOR = "OTHER"


def get_sector(ticker: str) -> str:
    """
    Map a raw ticker (with or without NSE suffix) to its NSE sector bucket.

    Unknown tickers fall back to DEFAULT_SECTOR so the pipeline keeps working;
    the SectorNeutralScaler in train_ranking.py will lump them into the global
    fallback scaler when per-sector sample count is thin.
    """
    base = (
        ticker.replace("_NS", "")
        .replace(".NS", "")
        .replace("-EQ", "")
        .strip()
        .upper()
    )
    return SECTOR_MAP.get(base, DEFAULT_SECTOR)
