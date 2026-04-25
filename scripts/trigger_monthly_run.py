from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import requests


def main() -> None:
    api_base = os.environ.get("RAMT_API_BASE", "http://127.0.0.1:8000/api")
    end_date_exclusive = os.environ.get(
        "RAMT_END_DATE_EXCLUSIVE", (date.today() + timedelta(days=1)).isoformat()
    )

    url = f"{api_base}/jobs/run-monthly"
    r = requests.post(url, json={"end_date_exclusive": end_date_exclusive}, timeout=30)
    r.raise_for_status()
    job_id = r.json().get("job_id")
    if not job_id:
        raise SystemExit("No job_id in response.")
    print(job_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise

