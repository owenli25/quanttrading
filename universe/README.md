# Universe Snapshots

Dated, versioned constituent snapshots for the factor-mining system's
default universe: the **union of current S&P 500 and Nasdaq-100 members**
(~518 large-cap US stocks as of 2026-08-21).

## Snapshot ID convention

Each snapshot carries a `universe_version`: the first 12 hex chars of the
SHA256 of its sorted ticker list. Agents (orchestrator provenance,
experiment controller, vn.py backtester) must reference this ID verbatim.

Current snapshot:

| Field | Value |
|---|---|
| `universe_version` | `538ced2e9326` |
| date | 2026-08-21 |
| count | 518 (416 SPX-only, 15 NDX-only, 87 in both) |
| files | `2026-08-21_spx_ndx.csv`, `latest.csv` |

## Regenerating

```bash
python generate_snapshot.py            # writes universe/<date>_spx_ndx.csv + latest.csv
```

Requirements: `pip install pandas requests lxml`.

Data sources: Wikipedia S&P 500 companies table (with GICS sectors);
official Nasdaq API (`api.nasdaq.com`) for Nasdaq-100 tickers.

## Known limitation (pre-declared)

Historical index membership is **not** reconstructed: backtesting this
current-constituent snapshot over history carries survivorship bias.
Per the agent profiles this is a declared system limitation, not a
blocking defect — but promoted candidates should be re-tested on a wider
snapshot (e.g. Russell 1000 current constituents) to confirm robustness,
and any hypothesis defined on index-inclusion events must not use this
universe at all.
