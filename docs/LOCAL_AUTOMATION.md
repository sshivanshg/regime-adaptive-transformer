# Local automation (monthly)

This project is designed to run locally with a FastAPI backend that exposes a monthly job endpoint.

## Prereqs

- Start the backend:

```bash
. .venv/bin/activate
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

## Manual trigger

```bash
. .venv/bin/activate
python scripts/trigger_monthly_run.py
```

Environment overrides:

- `RAMT_API_BASE` (default `http://127.0.0.1:8000/api`)
- `RAMT_END_DATE_EXCLUSIVE` (default today+1)

## launchd (macOS)

1) Copy the template and edit paths if needed:

`scripts/launchd/com.ramt.monthly.plist`

2) Load it:

```bash
launchctl load -w ~/Library/LaunchAgents/com.ramt.monthly.plist
```

3) Check status:

```bash
launchctl list | grep com.ramt.monthly
```

Logs will be written into `logs/` and artifacts into `results/runs/`.

