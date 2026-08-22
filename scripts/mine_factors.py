"""
mine_factors.py — GPU 因子挖掘 CLI

任务编排:
  数据(缓存parquet) -> 预处理 -> GPU特征面板 -> [每前瞻期] 遗传搜索 + Alpha101
  -> IC/ICIR/t/分层/换手 报告 -> 保存 output/mining/

用法:
  python scripts/mine_factors.py --n-exprs 400 --generations 2 --horizons 5 21
  python scripts/mine_factors.py --tickers NVDA AMD MU ... --start 2020-01-01 --seed 7
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import factor_lib as fl  # noqa: E402
from factor_mining.alphas101 import ALPHAS, build_base  # noqa: E402
from factor_mining.evaluate import ic_matrix, ic_summary, quantile_spread, turnover  # noqa: E402
from factor_mining.features import (FEATURES_ORDER, UNIVERSE, FeaturePanel,  # noqa: E402
                                    load_data, preprocess)
from factor_mining.search import FactorMiner, _sanitize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="GPU 因子挖掘")
    ap.add_argument("--tickers", nargs="+", default=UNIVERSE, help="标的列表 (默认100只)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--horizons", nargs="+", type=int, default=[5, 21], help="前瞻期(交易日)")
    ap.add_argument("--n-exprs", type=int, default=400, help="每代表达式数")
    ap.add_argument("--generations", type=int, default=2, help="遗传代数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=20, help="报告 top N")
    ap.add_argument("--no-cache", action="store_true", help="强制重新下载数据")
    args = ap.parse_args()

    print(f"设备: {fl.DEVICE} ({torch.cuda.get_device_name(0)})" if fl.DEVICE.type == "cuda"
          else f"设备: {fl.DEVICE}")
    print(f"标的: {len(args.tickers)} | 前瞻: {args.horizons} | 种群: {args.n_exprs} x {args.generations}代")

    # ---- 数据 ----
    t0 = time.perf_counter()
    frames = load_data(args.tickers, args.start, cache=not args.no_cache)
    frames = preprocess(frames["Close"], frames["Volume"],
                        {"Open": frames["Open"], "High": frames["High"], "Low": frames["Low"]})
    print(f"面板: {frames['Close'].shape[0]} 日 x {frames['Close'].shape[1]} 标的 "
          f"({frames['Close'].index[0].date()} ~ {frames['Close'].index[-1].date()}) "
          f"[{time.perf_counter()-t0:.0f}s]")

    panel = FeaturePanel(frames)
    features = panel.build_features()
    c, o, h, l, v = (panel.tensor(n) for n in ("Close", "Open", "High", "Low", "Volume"))
    base = build_base(c, o, h, l, v)
    print(f"GPU 特征: {len(features)} 个 ({time.perf_counter()-t0:.0f}s)")

    out_dir = ROOT / "output" / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    for h in args.horizons:
        print(f"\n========== 前瞻 {h} 日 ==========")
        # 前瞻收益排名
        fwd_df = frames["Close"].shift(-h) / frames["Close"] - 1.0
        fwd_t = torch.tensor(fwd_df.values.T, dtype=torch.float32, device=fl.DEVICE)
        fwd_rank = fl.cs_rank(torch.nan_to_num(fwd_t, nan=0.5))

        # ---- Alpha101 基线 ----
        print("[Alpha101]")
        names, stack = [], []
        for an, fn in ALPHAS.items():
            try:
                f = _sanitize(fn(base))
                if torch.isfinite(f).float().mean() > 0.5:
                    names.append(an)
                    stack.append(f)
            except Exception as ex:  # noqa: BLE001
                print(f"  skip {an}: {ex}")
        if stack:
            S = torch.stack(stack)
            ic = ic_matrix(S, fwd_rank)
            alpha_df = ic_summary(ic, names, panel.dates).sort_values("ICIR", ascending=False)
            print(f"  {len(names)} 个 alpha, top3: "
                  + ", ".join(f"{r.factor} ICIR={r.ICIR:+.3f}" for r in alpha_df.head(3).itertuples()))
        else:
            alpha_df = pd.DataFrame(columns=["factor", "mean_IC", "ICIR", "t_stat", "IC_positive_pct", "n_days"])

        # ---- 遗传搜索 ----
        print(f"[遗传搜索] 种群 {args.n_exprs} x {args.generations} 代 (seed={args.seed})")
        miner = FactorMiner(features, fwd_rank, panel.dates, FEATURES_ORDER)
        t1 = time.perf_counter()
        search_df, S_search = miner.genetic_search(n_pop=args.n_exprs,
                                                   generations=args.generations,
                                                   seed=args.seed)
        # OOM 修复: 精修变体不进报告, S 只保留 top60 对应行
        search_df = search_df.reset_index(drop=True)
        keep = search_df.head(60).index.to_numpy()
        S_search = S_search[keep].contiguous()
        search_df = search_df.iloc[keep]
        print(f"  搜索总耗时 {time.perf_counter()-t1:.1f}s, 精英表达式 {len(search_df)} 个")

        # ---- 合并报告 ----
        sd = search_df.drop(columns=[c for c in ("turnover", "corr_penalty", "fitness", "parent_id", "expr")
                                     if c in search_df.columns])
        merged = pd.concat([alpha_df.assign(source="alpha101"),
                            sd.assign(source="search")], ignore_index=True)
        merged = merged.sort_values("ICIR", ascending=False)
        # 补充分层 + 换手 (对 top 60)
        top60 = merged.head(60)
        sidx = {n: i for i, n in enumerate(search_df["factor"])}
        rows = []
        for n in top60["factor"]:
            if n in names:
                rows.append(_sanitize(ALPHAS[n](base)))
            else:
                rows.append(_sanitize(S_search[sidx[n]]))
        S_all = torch.stack(rows, dim=0)
        del rows
        q = quantile_spread(S_all, fwd_t, list(top60["factor"]))
        t = turnover(S_all, list(top60["factor"]))
        report = top60.merge(q, on="factor", how="left").merge(t, on="factor", how="left")

        print(f"\n----- Top {args.top} (前瞻 {h} 日) -----")
        show = report.head(args.top)[["factor", "source", "mean_IC", "ICIR", "t_stat",
                                      "IC_positive_pct", "turnover", "Q5_minus_Q1"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:.1f}"))

        # ---- 保存 ----
        report.to_csv(out_dir / f"report_h{h}_{stamp}.csv", index=False)
        show.to_csv(out_dir / f"top{h}_{stamp}.csv", index=False)
        # 保存 top 30 因子值 (parquet)
        top30 = report.head(30)
        vals = {}
        for name in top30["factor"]:
            if name in names:
                idx = names.index(name)
                vals[name] = pd.DataFrame(stack[idx].cpu().numpy().T,
                                          index=panel.dates, columns=panel.tickers)
            else:
                vals[name] = pd.DataFrame(S_search[sidx[name]].cpu().numpy().T,
                                          index=panel.dates, columns=panel.tickers)
        pd.concat(vals, axis=1).to_parquet(out_dir / f"top30_values_h{h}_{stamp}.parquet")
        print(f"  已保存 -> {out_dir}/ (report_h{h}, top{h}, top30_values_h{h})")

    print("\n完成。建议下一步: 对 top 因子做中性化/合成 (IC加权), 或设 cron 每日挖掘。")


if __name__ == "__main__":
    main()
