# Factor Mining Pipeline (ROCm GPU)

GPU-accelerated cross-sectional factor mining suite: genetic-programming
factor search + Alpha101 baselines, batch-evaluated on AMD ROCm via PyTorch.

Validated on RX 9070 XT / ROCm 7.2.4 / WSL2 (torch 2.15 rocm7.2).
Reference deployment: `D:\quant` (env via `env.sh`, data cache in
`data/ohlcv_*.parquet`, output in `output/mining/`).

## Layout

```
scripts/
  factor_lib.py          — device/panel helpers (cs_rank, rolling ops)
  mine_factors.py        — CLI orchestration: data -> GPU features ->
                           [Alpha101 + GP search] -> IC report
  factor_mining/
    features.py          — FeaturePanel: yfinance load, preprocessing,
                           23 GPU feature builders, UNIVERSE
    expressions.py       — AST expression trees (tuple form), Evaluator
                           with subtree memo
    evaluate.py          — batched Spearman IC (einsum), quantile spread,
                           turnover; chunked ranking; rank reuse
    search.py            — genetic search v2 (see below)
    alphas101.py         — 29 GPU-vectorized WorldQuant Alpha101 factors
```

## Usage

```bash
python scripts/mine_factors.py --n-exprs 400 --generations 4 \
    --horizons 5 21 --seed 42
```

Outputs per horizon: `report_h{h}.csv` (full stats), `top{h}.csv`,
`top30_values_h{h}.parquet` (factor value panels).

## Genetic Search v2

Operators are codified in `.github/agents/factor-experiment-controller.agent.md`
(Search Operator Discipline / Memory Safety Rules). Summary:

- Mild mutation (12% subtree reset) with window/feature micro-mutation;
  tournament selection k=5.
- Composite fitness `ICIR − 0.25·turnover − 0.3·corr_penalty`
  (λ, μ pre-registered).
- Bucketed elite pool (vol/volume/momentum/meanrev, top-8 each, cap 60).
- Two-stage: GP coarse search → window/constant grid refinement of top-40
  elites (≤64 variants/tree, batched GPU eval, best variant per parent kept).

### Memory safety (each rule traces to a real OOM kill)

1. Never materialize combinatorial grids (`len(grid)^n_params` explodes);
   sample parameter indices when the theoretical space exceeds the cap.
2. Discard batch tensors immediately after per-batch reduction
   (`gc.collect()` + `torch.cuda.empty_cache()` between batches).
3. Crop returned panels to consumed rows (~97% of refinement rows unused).
4. Batched GPU metrics only — no per-factor `.cpu().numpy()` loops.
5. Rank reuse: turnover consumes the centered ranks from `ic_matrix(...,
   want_ranks=True)` instead of re-running the double argsort; ranking is
   chunked (128 rows) to bound transient memory.

## Known limitations

- Constant-subtree degenerate variants (identical ICIR across remappings)
  require structure-signature dedup — planned, not yet implemented.
- Survivorship bias from current-constituent universes is pre-declared
  (see `universe/README.md`); promoted candidates need a wider-universe
  sensitivity check.
