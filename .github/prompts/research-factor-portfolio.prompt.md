---
name: "Research Factor Portfolio"
description: "Assemble individually promoted factors into a pre-registered portfolio and evaluate it through the Assembly Gate: diversification-first member selection, net-of-cost additivity tests, attribution, and vn.py portfolio backtesting."
argument-hint: "Optional overrides: candidate members, weighting scheme, correlation cap, rebalance schedule, capital, dates, costs."
agent: "Factor Mining Orchestrator"
---
Assemble and evaluate a factor portfolio from individually promoted candidates. Treat user-supplied arguments as explicit overrides registered before any result is observed. Default to the discovery tier. Do not access the final holdout.

## Objective

Determine whether a combination of promoted factors produces a net-of-cost, deflated portfolio Sharpe that is **additive** — i.e. beats its best single member and an equal-weight naive blend of the same members. Diversification, not member Sharpe, drives selection: the value of a member is the correlation it does not already have.

## Member Eligibility

- Only candidates individually promoted through the ROCm and vn.py gates, with immutable signal artifacts and recorded `turnover_control`.
- Rejected, failed, or unpromoted candidates are inadmissible — including "promising ones to test together later".
- Prefer members whose pairwise residual-return correlations (market- and industry-residual) are below the pre-registered cap (default 0.5). If fewer than two eligible members clear the cap, return `blocked` with the correlation matrix instead of loosening the cap.

## Pre-Registered Assembly Specification

Register all of the following before evaluating anything:

1. Member list and each member's role (return driver, risk diversifier, regime hedge).
2. Weighting scheme: equal-weight, IC-IR weighted, or inverse-variance. No other schemes this run.
3. Residual-correlation cap and the correlation window used to estimate it.
4. Conflict-netting rule for same-symbol offset signals (net by weight; report netting counts).
5. Rebalance schedule; each member inherits its own pre-registered `turnover_control` (buffer, smoothing) unchanged.
6. Cost tier: large-cap floor 8bp per side all-in unless the user documents lower.
7. Combination budget: at most 5 distinct assembly specifications per run; every one counts toward family-wise trial accounting.

## Evaluation

- Run each specification through the vn.py backtester in portfolio mode with per-member attribution, correlation matrix, and conflict-netting counts.
- Apply the Assembly Gate: beat best member AND equal-weight naive blend, net of costs and deflated; otherwise `non_additive`.
- Report regime stability of the increment (the portfolio advantage must not live in a single regime).
- If no specification is additive, report that honestly — a negative result with full attribution is a valid outcome, not a failure of the run.

## Required Next Action

If additive: name the promoted portfolio artifact and the paper-trade recommendation path. If not: name the binding constraint (correlations, costs, or regime concentration) and the single most informative follow-up.
