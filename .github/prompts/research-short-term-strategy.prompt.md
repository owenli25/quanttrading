---
name: "Research Short-Term Strategy"
description: "Research an auditable 1-10 trading-day US equity strategy using liquid high-beta stocks, technical factors, macro regimes, ROCm evaluation, and vn.py backtesting."
argument-hint: "Optional overrides: capital, dates, holding period, universe, long-only/long-short, data paths, costs, and risk limits."
agent: "Factor Mining Orchestrator"
---
Run the complete factor-mining pipeline for a short-term US equity strategy. Treat any arguments supplied by the user with this prompt as explicit overrides. Pre-register all choices before inspecting performance, and default to the discovery tier. Do not access the final holdout.

## Objective

Find a tradable cross-sectional signal for liquid, high-beta US common stocks that predicts market- and industry-residual returns over 1, 3, 5, and 10 trading days. High beta defines the opportunity set and risk exposure; it is not alpha by itself.

## Default Universe

- Use point-in-time US common stocks and retain delisted securities.
- Exclude ETFs, ADRs, funds, preferred shares, OTC securities, SPACs, and securities with unresolved type metadata.
- Require point-in-time float market capitalization of at least USD 2 billion.
- Require an unadjusted tradable price of at least USD 5.
- Require 20-day average dollar volume of at least USD 50 million and at least 252 trading days of history.
- Limit each order to at most 1% of 20-day average dollar volume.
- Build the high-beta subset from eligible stocks using lagged, robust 20-, 60-, and 120-day beta estimates versus SPY, downside beta, estimation error, and beta stability.
- Use 60-day beta as the primary horizon. Require beta above 1.2 and select approximately the top 20% after eligibility filters.
- Neutralize candidate stock scores against industry and log float market capitalization. Report any remaining market beta, size, volatility, and sector exposure.

If point-in-time float shares, delisting history, universe membership, or security-type metadata is unavailable, identify the exact missing input and stop before executable evaluation rather than substituting current data.

## Hypothesis Families

Research these mechanisms separately before considering an ensemble:

1. Residual momentum over 3, 5, and 10 days, including acceleration after removing SPY and industry returns.
2. Intraday strength using close location value, open-to-close return, VWAP displacement when intraday data exists, and persistence of strong closes.
3. Volume and liquidity shocks using dollar-volume surprise, turnover surprise, price-volume confirmation, and changes in Amihud illiquidity.
4. Breakout quality using 20-day breakout distance, ATR expansion, and trend efficiency rather than raw volatility alone.
5. Short-term reversal risk using overnight gaps, distance from the 10-day mean in ATR units, upper shadows, and extreme 1-day residual returns.

For each family, state the economic or behavioral mechanism, expected direction, failure regime, required fields, publication lag, and a falsification test. Generate at most five economically distinct hypotheses and at most five DSL variants per hypothesis. Do not present cosmetic window changes as new hypotheses.

## Macro Regime Layer

Keep macro variables outside the stock-ranking expression. They may change only total exposure, portfolio beta target, or risk budget unless a separate cross-sectional interaction is pre-registered and approved.

Begin with timely daily proxies:

- SPY trend and 5-/20-day return.
- VIX level, 5-day change, and term structure when point-in-time data exists.
- Market breadth, including the share of eligible stocks above their 20- and 50-day averages.
- Credit and risk appetite using HYG/LQD and IWM/SPY.

Define `Risk-On`, `Neutral`, `Risk-Off`, and `Rebound` states. Compare the unchanged stock signal under four pre-registered variants: no macro layer; trend plus VIX; breadth plus credit; and the complete macro layer. Retain a macro layer only if it improves out-of-sample cost-adjusted risk or drawdown and remains stable near its thresholds.

## Portfolio And Execution

- Compare long-only top-decile, beta-controlled long-only, and market- or beta-neutral long-short portfolios when borrow data permits.
- Use a 3- to 5-day primary holding period, with 1- and 10-day horizons as robustness checks.
- Generate signals after the US close and trade no earlier than the next session's open or a pre-registered VWAP window.
- Cap a single-stock position at 2% and a sector at 20% unless the user supplies stricter limits.
- Include commissions, bid-ask spread, market impact, SEC and FINRA fees where applicable, borrow cost, rejected orders, and cash drag.
- Compare against SPY, an equal-weight high-beta basket, residual momentum, and short-term reversal baselines.
- Report overnight and intraday PnL separately when data supports it.

## Validation

- Use chronological discovery and validation splits with purging or embargo where overlapping labels require it.
- Report IC, ICIR, quantile monotonicity, net return, Sharpe, drawdown, turnover, capacity, exposure, and stability by year and regime.
- Test nearby windows, universe thresholds, execution delays, and cost assumptions without choosing a single isolated optimum.
- Reject strategies whose performance depends only on the smallest stocks, event-day gaps, unavailable borrow, same-bar execution, or a narrow parameter point.
- Record every attempted hypothesis and variant, including failures and duplicates.

Run the available stages automatically. If data, DSL operators, the ROCm environment, or the pinned vn.py environment is missing, stop at the corresponding gate and return the precise setup requirements. Do not fabricate a completed backtest.
