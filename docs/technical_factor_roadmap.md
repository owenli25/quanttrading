# Technical Factor Research Roadmap

Scope: cross-sectional technical signals on the dated S&P 500 ∪ Nasdaq-100
constituent snapshot (`universe/latest.csv`, universe_version recorded at run
start). Daily frequency, market- and industry-residual returns, macro layer
separate from stock ranking unless a pre-registered cross-sectional
interaction passes Critic review.

This roadmap guides hypothesis selection for `Factor Research`. Priority
order reflects expected net-of-cost information per unit of implementation
complexity on ~518 liquid large caps, where single raw-factor gross IC is
typically 0.02–0.05; the realistic alpha pool is interactions, regime
conditioning, and turnover control, not novel single factors.

---

## 1. Volatility Families

### 1.1 Estimator upgrades (highest priority, lowest cost)
Replace close-close realized vol with range estimators registered from
existing OHLC fields:

| Estimator | Inputs | Why |
|---|---|---|
| Parkinson | H, L | ~5x efficiency of close-close |
| Garman-Klass | O, H, L, C | ~7.4x; robust intraday |
| Yang-Zhang | O, H, L, C | only estimator handling overnight gaps AND drift without bias — most relevant for US large caps where overnight gaps carry a large share of total variance |
| Downside semi-deviation | C | asymmetry; low downside vol is a cleaner defensive trait than total vol |

### 1.2 Volatility structure
- **Term slope**: vol(10d)/vol(60d); >1 = expanding. Expected negative residual returns ahead (clustering + overreaction decay).
- **Vol-of-vol**: std of the rolling vol series; uncertainty pricing.
- **Idiosyncratic vol**: residual (market+industry removed) realized vol, not total vol — aligns with the pipeline's neutralization step; strongest-documented low-vol variant.
- **Up/down semi-variance ratio**: healthy tape = upside variance dominates.

### 1.3 Regime conditioning (uses the existing macro layer)
Low-vol alpha is state-dependent: strongest in Risk-Off, reverses early in
Rebound (high-beta catch-up). Pre-register grouped IC by macro state;
holding period should match vol-clustering half-life (~10–20 trading days),
not the 20-day default borrowed from momentum.

## 2. Volume / Flow Families

### 2.1 Comparability precondition (required before any volume factor)
Raw volume is not comparable across names. Normalize first:
1. Turnover: `volume / shares_out` (register `shares_out`; slow-moving, no PIT burden).
2. Self z-score: `(V − mean(V,60d)) / std(V,60d)` — each name versus itself.

### 2.2 Core factors

| Factor | Daily construction | Mechanism | Horizon |
|---|---|---|---|
| Amihud illiquidity | `mean(abs(ret)/dollar_vol, 21d)` | liquidity premium; gradient persists even within large caps | 1–3m |
| Price-volume divergence | `−corr(ret, dV, 20d)` | rising on shrinking volume = exhaustion | 5–20d |
| Momentum confirmed by volume | `sign(mom_20d) × V_z` | interaction: only volume-confirmed momentum persists | 10–40d |
| CLV money flow (CMF/MFI class) | `mean(((C−L)−(H−C))/(H−L) × V_z, 20d)` | accumulation/distribution proxy | 10–60d |
| Volume-dry-up reversal | low V_z after large drop | liquidity-provider compensation; fits the 1–10d short-horizon tier | 1–10d |
| Abnormal-volume day density | share of last 60d with abs(V_z)>2 | information-event density; uncertainty premium | 1–3m |

### 2.3 Dual identity warning
Every volume factor carries two distinct claims — pick one and pre-register
it: (a) *risk characteristic* (liquidity/capacity, premium direction), or
(b) *sentiment/confirmation signal* (price-volume relation). Control the
other in `baseline_exposures_to_control` (Amihud and reversal correlate).

## 3. Event-Driven And Conditional Families

These require pre-registered cross-sectional interactions (the Critic-gate
path reserved for them). All are computable from OHLCV alone.

### 3.1 Short-term pullback within an established trend
- **Shape**: trend filter × short-term weakness interaction, e.g.
  `rank(trend_strength) × rank(short_term_weakness)` where trend strength =
  residual momentum 3–12m or price vs 200d MA, and weakness = −ret_5d,
  low RSI(2..5), or drawdown below the 20d MA.
- **Mechanism**: liquidity provision to forced/impatient sellers inside a
  persistent drift; harvests the well-documented short-term reversal only
  where the medium-term drift protects against trend-end risk.
- **Horizon**: 5–20 trading days. **Variant axes**: trend definition,
  weakness depth threshold, entry timing (close-of-signal vs next open).
- **Pitfall**: must exclude earnings-window days or control for the
  announcement effect — much "pullback buying" is actually PEAD harvesting.

### 3.2 Breakout after volatility contraction (squeeze)
- **Shape**: binary compression flag × post-expansion direction, e.g.
  `squeeze_flag × sign(forward_breakout)` operationalized as
  `1[percentile(vol_ratio_10_60, 252d) < 0.1]` interacted with the sign of
  the breach of the N-day high/low band.
- **Mechanism**: volatility clustering — compression precedes expansion;
  coiled ranges resolve directionally more often than randomly, and the
  resolution direction inherits local order-flow imbalance.
- **Horizon**: 5–20 days after resolution. **Variant axes**: compression
  metric (BBW percentile vs GK/YZ ratio vs ATR ratio), band length
  (20/55d), resolution window.
- **Evaluation note**: this is an event-conditioned factor — report
  conditional forward returns and hit rates, not unconditional IC alone;
  flag base-rate sparsity (how many names are in squeeze state per rebalance)
  so capacity is honestly assessed.

### 3.3 Confirmed breakout vs failed breakout (trap detection)
- **Shape**: two complementary hypotheses on 20d-high/lows breaches:
  - *Confirmed*: `breakout_flag × rank(V_z)` — high-volume breakouts continue (order-flow validation of new levels).
  - *Trap*: `breakout_flag × rank(−V_z)` (and failed-breach re-entry) — low-volume breaks revert; short side subject to borrow constraints, prefer expressing as avoidance/underweight rather than explicit shorts.
- **Mechanism**: breakouts without participation lack the flow to defend new prices; traps monetize stop-run overreaction.
- **Horizon**: 5–15 days. **Variant axes**: breach threshold (1.0x/1.5x band), V_z window (20/60d), gap-vs-intraday break distinction.
- **Pitfall**: survivorship-safe here (no delisting dependence), but watch
  look-ahead in band construction — bands must use data strictly before the
  decision timestamp.

### 3.4 Market-breadth risk switch (macro regime layer, not ranking)
- **Definition**: breadth state variables computed across the universe:
  - % of constituents above their 50d/200d MA;
  - 21d advance-decline line slope;
  - equal-weight vs cap-weight universe spread (RSP/SPY analogue computed
    internally — no external ETF needed);
  - % of members making 20d lows.
- **Usage**: feeds the existing `macro_regime_layer` as additional
  Risk-On/Risk-Off/Neutral/Rebound inputs. Per standing rules, breadth may
  change total exposure / beta target / risk budget only; a breadth ×
  factor-exposure interaction (scaling specific factor sleeves by state) is
  admissible solely through the pre-registered interaction path.
- **Pre-register the ablation battery** exactly as the orchestrator defines:
  none vs trend+VIX vs breadth+credit vs complete. Retain the layer only if
  out-of-sample cost-adjusted risk/drawdown improves and thresholds are stable.
- **Why breadth earns its place**: breadth deterioration typically leads
  index-level risk-off while individual-stock signals still read neutral;
  it is the cheapest available aggregate of "how many names are actually
  participating".

## 4. Interaction And Nonlinearity Backlog

| Hypothesis | Form | Rationale |
|---|---|---|
| Momentum × idio-vol | `rank(mom_12m) × rank(−idio_vol_60d)` | quality-momentum; hedges momentum crashes |
| Reversal × volume | `rank(−ret_5d) × rank(volume_z)` | only capitulation-with-volume reverts |
| Low-vol × macro state | low-vol sleeve active in Risk-Off only | upgrade macro layer from position sizing to factor-exposure scaling (needs interaction approval) |
| Trend × pullback | §3.1 above | |
| Squeeze × direction | §3.2 above | |

DSL operator requirements to support everything above: `rolling_std`,
`rolling_skew`, `rolling_kurt`, `rolling_corr`, `semi_std(up/down)`, `ema`,
`winsorize_mad`, `div`, `sign`, `indicator(cond)` (binary), `max/min`
rolling bands. Winsorization/MAD clipping is mandatory for vol/volume
factors (heavy tails dominate IC otherwise); EMA smoothing (3–5d) of raw
volume signals is a standard generated variant, chosen for turnover-net
Sharpe improvement, reported alongside the unsmoothed version.

## 5. Evaluation Additions

- Grouped IC by macro state (supports §1.3, §3.4).
- Turnover-vs-net-Sharpe frontier (quantifies EMA-smoothing benefit).
- Event-family diagnostics: conditional forward-return curves, state
  occupancy counts, hit rates (§3.2, §3.3 are event-conditioned).
- Multiple-testing discipline and survivorship sensitivity checks per the
  Experiment Controller profile apply to every family here.

## 6. Priorities

| # | Item | Tier | Why |
|---|---|---|---|
| 1 | Yang-Zhang/Parkinson idio-vol | discovery→validation | best-documented, thinnest implementation |
| 2 | Volume normalization + Amihud + divergence | discovery | fills the volume gap; gradient survives in large caps |
| 3 | §3.1 trend-pullback, §3.3 breakout/trap | discovery | conditional interactions carry the remaining large-cap alpha pool; short horizon diversifies timing |
| 4 | §3.2 squeeze breakout | discovery | event-conditioned; strong mechanism, needs honest base-rate reporting |
| 5 | §3.4 breadth risk switch | validation-layer work | improves risk management rather than ranking IC; evaluate through the macro ablation battery |
| 6 | Interaction backlog (§4) | rolling | main compounding source once singles are registered |
