#!/usr/bin/env python
"""Generate the dated S&P 500 ∪ Nasdaq-100 constituent snapshot.

Produces universe/<YYYY-MM-DD>_spx_ndx.csv plus updates universe/latest.csv.
The snapshot ID is the SHA256 of the sorted ticker list, truncated to 12 hex
chars; agents must reference this ID verbatim as `universe_version`.

Usage:
    python generate_snapshot.py [--out-dir PATH]

Requirements: pandas, requests, lxml (pip install pandas requests lxml).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from datetime import date, datetime, timezone

import pandas as pd
import requests

WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (factor-mining-universe-snapshot/1.0)"
}

SPX_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def fetch_spx() -> pd.DataFrame:
    html = requests.get(SPX_URL, headers=WIKI_HEADERS, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if "Symbol" in t.columns)
    out = pd.DataFrame(
        {
            "ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
            "name": t["Security"].astype(str),
            "sector": t.get("GICS Sector", pd.Series([""] * len(t))).astype(str),
            "industry": t.get("GICS Sub-Industry", pd.Series([""] * len(t))).astype(str),
        }
    )
    out["index"] = "SPX"
    return out


NDX_API = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"


def fetch_ndx() -> pd.DataFrame:
    """Fetch Nasdaq-100 constituents from the official Nasdaq API.

    The Wikipedia table no longer carries ticker symbols; the official API is
    the authoritative source anyway.
    """
    h = dict(WIKI_HEADERS)
    h.update({"Accept": "application/json", "Accept-Language": "en-US,en;q=0.9"})
    r = requests.get(NDX_API, headers=h, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["data"]["rows"]
    out = pd.DataFrame(
        {
            "ticker": [str(row["symbol"]).replace(".", "-") for row in rows],
            "name": [str(row.get("companyName", "")) for row in rows],
            "sector": [""] * len(rows),
            "industry": [""] * len(rows),
        }
    )
    out["index"] = "NDX"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Directory for snapshots (default: alongside this script)")
    args = ap.parse_args()
    out_dir = args.out_dir or "."

    spx = fetch_spx()
    ndx = fetch_ndx()

    combined = pd.concat([spx, ndx], ignore_index=True)
    dupes = combined.duplicated(subset=["ticker"], keep="first")
    union = combined[~dupes].copy()
    union["in_both_indices"] = False
    both_mask = combined[dupes]["ticker"].unique()
    union.loc[union["ticker"].isin(both_mask), "in_both_indices"] = True

    tickers = sorted(union["ticker"].tolist())
    snapshot_id = hashlib.sha256("\n".join(tickers).encode()).hexdigest()[:12]

    today = date.today().isoformat()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    union = union.sort_values("ticker").reset_index(drop=True)
    header = (
        f"# universe_version={snapshot_id}\n"
        f"# snapshot_date={today}\n"
        f"# captured_at_utc={now_utc}\n"
        f"# definition=union of current S&P 500 and Nasdaq-100 constituents\n"
        f"# source_spx={SPX_URL}\n"
        f"# source_ndx={NDX_API}\n"
        f"# count={len(union)}\n"
        f"# note=survivorship bias is a pre-declared system limitation; "
        f"historical index membership is NOT reconstructed\n"
    )

    csv_path = f"{out_dir}/{today}_spx_ndx.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write(header)
        union.to_csv(f, index=False, quoting=csv.QUOTE_MINIMAL)

    # latest.csv: copy without the date-specific filename for stable referencing
    latest_path = f"{out_dir}/latest.csv"
    with open(csv_path, encoding="utf-8") as src, open(latest_path, "w", encoding="utf-8", newline="") as dst:
        dst.write(src.read())

    print(f"universe_version={snapshot_id}")
    print(f"snapshot_date={today}")
    print(f"count={len(union)}")
    print(f"spx_only={int((~union['in_both_indices'] & (union['index'] == 'SPX')).sum())}")
    print(f"ndx_only={int((~union['in_both_indices'] & (union['index'] == 'NDX')).sum())}")
    print(f"in_both={int(union['in_both_indices'].sum())}")
    print(f"written={csv_path}")
    print(f"written={latest_path}")

    missing_meta = int(((union["sector"] == "") & ~union["in_both_indices"]).sum())
    if missing_meta:
        print(f"warning: {missing_meta} rows lack sector metadata", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
