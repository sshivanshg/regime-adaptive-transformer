nifty100_nse_survivorship_proxy.txt — one Yahoo ticker per line (NIFTY 100 constituents from the
current NSE CSV at generation time). Not point-in-time for 2010–2015; use for large-cap proxy only.
Regenerate: python scripts/fetch_nifty200.py --list-only --index nifty100 --save-symbol-list scripts/universe/nifty100_nse_survivorship_proxy.txt
