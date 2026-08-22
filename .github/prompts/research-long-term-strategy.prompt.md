---
name: "Research Long-Term Strategy"
description: "Research an auditable 1-12 month US equity multi-factor strategy on the S&P 500 / Nasdaq-100 snapshot using technical factors, macro regimes, ROCm evaluation, and vn.py backtesting."
argument-hint: "Optional overrides: capital, dates, holding horizon, universe, benchmark, data paths, factor families, costs, and risk limits."
agent: "Factor Mining Orchestrator"
---
Run the complete factor-mining pipeline for a long-horizon US equity strategy. Treat any arguments supplied by the user with this prompt as explicit overrides. Pre-register all choices before inspecting performance, and default to the discovery tier. Do not access the final holdout.

## Objective

Find a capacity-aware cross-sectional strategy that predicts market- and industry-residual returns over 1, 3, 6, and 12 months. Prefer persistent economic mechanisms and low turnover over fragile short-term improvements.

## Default Universe

- Use the dated S&P 500 ∪ Nasdaq-100 constituent snapshot (universe/latest.csv, versioned by SHA256 ID); do not reconstruct historical index membership. Survivorship bias from current-constituent backtesting is a pre-declared limitation — promoted candidates must pass the wider-universe sensitivity check.
- Exclude ETFs and ADRs present in the snapshot.
- Require an unadjusted tradable price of at least USD 5 at signal time.
- Require 60-day average dollar volume of at least USD 20 million and at least 504 trading days of price history within the snapshot.
- Use current sector classifications from the snapshot; treat them as static over the backtest window and report any classification-change sensitivity for the longest runs.
- Neutralize candidate scores against industry unless the hypothesis explicitly pre-registers an exposure.
- Report results separately by size bucket computed within the snapshot and reject signals supported only by the smallest bucket.

If the universe snapshot, price/volume history, or sector metadata is unavailable, identify the exact missing input and stop before executable evaluation rather than substituting unversioned data.

## Hypothesis Families

Research these technical mechanisms separately before testing a diversified ensemble:

1. Medium-term momentum using 12-minus-1-month residual momentum, 6-minus-1-month momentum, trend consistency, and industry-relative strength.
2. Volatility structure using Yang-Zhang or Parkinson idiosyncratic volatility, volatility term slope, vol-of-vol, and downside semi-deviation.
3. Volume and liquidity using Amihud illiquidity on normalized volume, price-volume divergence, turnover z-score dynamics, and abnormal-volume day density.
4. Trend-with-pullback interactions: short-term weakness inside persistent medium-term drift, gated by trend strength (docs/technical_factor_roadmap.md §3.1).
5. Defensive characteristics using beta, drawdown resilience, up/down semi-variance ratio, and distance-from-high structure.
6. Seasonality and calendar structure using month-of-year and turn-of-month residual patterns with multiple-testing discipline applied.

Fundamental families (value, quality, growth, fundamental revisions) are out of scope for this pipeline's default data layer; propose them only if a documented source vintage is supplied, and they require explicit user authorization.

For each family, state the economic mechanism, expected direction, decay horizon, crowding risk, required fields, and falsification criteria. Generate at most six economically distinct hypotheses and at most four DSL variants per hypothesis. Test individual factors before combining them so ensemble gains can be attributed under the Assembly Gate.

## Data Timing

- Technical fields (price, volume, volatility) are complete at the close; signals computed after the close trade no earlier than the next session.
- If any non-technical field is authorized, use its original public release timestamp, not the fiscal-period end date; never backfill revised values into earlier dates.
- Lag sector classification to the snapshot date; do not imply intraperiod reclassification.

## Macro Regime Layer

Keep macro variables outside the stock-ranking expression. They may adjust only total exposure, portfolio beta target, factor risk budget, or defensive tilt unless a separate cross-sectional interaction is pre-registered and approved.

Study:

- SPY trend and market breadth.
- VIX level and term structure.
- Credit conditions using high-yield spreads or point-in-time tradable proxies such as HYG/LQD.
- Nominal and real yield levels, changes, and curve slope.
- DXY and financial-conditions or liquidity proxies when historical vintages are available.

Define interpretable growth, inflation, liquidity, and risk-appetite regimes without optimizing a large state space. Compare the unchanged stock signal under four pre-registered variants: no macro layer; trend plus VIX; breadth plus credit; and the complete macro layer. Retain the macro layer only when it improves out-of-sample cost-adjusted risk or drawdown and is robust to threshold and release-lag changes.

## Portfolio And Execution

- Use monthly rebalancing as the primary schedule and quarterly rebalancing as a turnover robustness check.
- Compare long-only top-quintile, benchmark-aware long-only, and market- or beta-neutral long-short portfolios when borrow history permits.
- Cap a single-stock position at 2%, a sector at 20%, and each order at 1% of 60-day average dollar volume unless the user supplies stricter limits.
- Control market beta, industry, size, volatility, and unintended style exposures.
- Include commissions, spread, market impact, SEC and FINRA fees where applicable, borrow cost, dividends, corporate actions, and cash drag.
- Compare against SPY, an eligible-universe value-weighted benchmark, and simple value, quality, and 12-minus-1 momentum baselines.

## Validation

- Use chronological discovery and validation periods spanning multiple market and rate regimes.
- Report IC, ICIR, quantile monotonicity, net return, Sharpe, drawdown, turnover, capacity, exposure, and stability by year, sector, size, and macro regime.
- Measure factor decay at 1, 3, 6, and 12 months and test monthly versus quarterly implementation.
- Test nearby definitions, lags, winsorization levels, universe thresholds, and cost assumptions without selecting an isolated optimum.
- Compare equal-weighted, value-weighted, and risk-controlled construction to identify small-cap or concentration dependence.
- Penalize high correlation with registered factors, excessive complexity, unstable signs, and gains that disappear after realistic publication lags.
- Record every attempted hypothesis and variant, including failures and duplicates.

Run the available stages automatically. If the universe snapshot, DSL operators, the ROCm environment, or the pinned vn.py environment is missing, stop at the corresponding gate and return the precise setup requirements. Do not fabricate a completed backtest.
