---
name: "Research Long-Term Strategy"
description: "Research an auditable 1-12 month US equity multi-factor strategy using point-in-time fundamentals, macro regimes, ROCm evaluation, and vn.py backtesting."
argument-hint: "Optional overrides: capital, dates, holding horizon, universe, benchmark, data paths, factor families, costs, and risk limits."
agent: "Factor Mining Orchestrator"
---
Run the complete factor-mining pipeline for a long-horizon US equity strategy. Treat any arguments supplied by the user with this prompt as explicit overrides. Pre-register all choices before inspecting performance, and default to the discovery tier. Do not access the final holdout.

## Objective

Find a capacity-aware cross-sectional strategy that predicts market- and industry-residual returns over 1, 3, 6, and 12 months. Prefer persistent economic mechanisms and low turnover over fragile short-term improvements.

## Default Universe

- Use point-in-time US common stocks and retain delisted securities.
- Exclude ETFs, ADRs, funds, preferred shares, OTC securities, SPACs, and securities with unresolved type metadata.
- Require point-in-time float market capitalization of at least USD 2 billion.
- Require an unadjusted tradable price of at least USD 5.
- Require 60-day average dollar volume of at least USD 20 million and at least 504 trading days of price history.
- Use historical sector classifications, shares outstanding, float shares, corporate actions, and universe membership as known on each date.
- Neutralize candidate scores against industry and log float market capitalization unless the hypothesis explicitly pre-registers an exposure.
- Report results separately for large-, mid-, and smaller-cap eligible stocks and reject signals supported only by the smallest bucket.

If point-in-time fundamentals, original release timestamps, restatement history, delisting history, or historical classifications are unavailable, identify the exact missing input and stop before executable evaluation rather than using today's revised values.

## Hypothesis Families

Research these mechanisms separately before testing a diversified ensemble:

1. Value using earnings yield, free-cash-flow yield, EBITDA or operating-profit yield, and shareholder yield with sector-aware comparisons.
2. Quality using profitability, gross profitability, ROIC, accruals, leverage, earnings stability, and balance-sheet improvement.
3. Growth and investment using sustainable revenue or earnings growth, asset growth, capital expenditure efficiency, and profitability-adjusted investment.
4. Fundamental revisions using earnings surprises, analyst estimate revisions, guidance changes, and post-earnings drift when timestamped data exists.
5. Medium-term momentum using 12-minus-1-month residual momentum, 6-minus-1-month momentum, trend consistency, and industry-relative strength.
6. Defensive characteristics using beta, idiosyncratic volatility, downside risk, drawdown resilience, and balance-sheet strength.

For each family, state the economic mechanism, expected direction, decay horizon, crowding risk, required fields, actual publication lag, and falsification criteria. Generate at most six economically distinct hypotheses and at most four DSL variants per hypothesis. Test individual factors before combining them so ensemble gains can be attributed.

## Data Timing

- Use the original public release timestamp, not the fiscal-period end date, to make a fundamental observation available.
- Apply a conservative additional processing lag when exact intraday release time is unavailable.
- Preserve historical vintages and restatements. Never backfill revised statements into earlier dates.
- Lag index membership, sector classification, shares, float, and analyst data to their actual availability.
- Align monthly and quarterly data without treating repeated stale observations as independent daily information.

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

Run the available stages automatically. If point-in-time data, DSL operators, the ROCm environment, or the pinned vn.py environment is missing, stop at the corresponding gate and return the precise setup requirements. Do not fabricate a completed backtest.
