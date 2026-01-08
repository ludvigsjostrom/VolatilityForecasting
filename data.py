#!/usr/bin/env python3
"""
Binance Futures (Tardis) BTC data downloader for a range of days.

Downloads:
  - quotes (top-of-book best bid/ask updates)

Install:
  pip install tardis-dev

Auth:
  export TARDIS_API_KEY="TD.xB2cKE-H8vebXkxh.5Ku-HKy4iABamYB.QNmqwwShurgHOQ4.CzriLdnTigVi2rL.Dk5YzL48UA--ULB.mfEh"
  # Without API key, only the first day of each month is downloadable per Tardis docs.

Run:
  python data.py --start 2025-11-15 --end 2025-11-20 --out ./tardis_binance_btc
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

from tardis_dev import datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date (inclusive), YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date (inclusive), YYYY-MM-DD")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance Futures instrument, default BTCUSDT")
    p.add_argument("--out", default="./tardis_binance_btc", help="Output directory")
    p.add_argument("--api-key", default=None, help="Tardis API key (or set TARDIS_API_KEY env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    if end_date < start_date:
        raise SystemExit("ERROR: --end must be >= --start")

    # Tardis uses exclusive end date, so add 1 day
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    api_key = args.api_key or os.environ.get("TARDIS_API_KEY", "")
    if not api_key:
        print("WARNING: No API key provided. Only sample days (first day of each month) may download without a key.")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_types = ["quotes"]

    num_days = (end_date - start_date).days + 1
    print(f"Downloading {args.symbol} from {from_date} to {end_date} ({num_days} days) from binance-futures")
    print(f"Data types: {data_types}")
    print(f"Output directory: {out_dir}")

    datasets.download(
        exchange="binance-futures",
        data_types=data_types,
        from_date=from_date,
        to_date=to_date,
        symbols=[args.symbol],
        api_key=api_key,
        download_dir=str(out_dir),
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
