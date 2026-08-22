import ast
import pathlib

p = pathlib.Path('/mnt/d/quant/scripts/factor_mining/search.py')
s = p.read_text(encoding='utf-8')

# 2) evaluate_with_fitness: 退化检测 (注意此文件 fitness 计算在检测前已存在,
#    实际顺序: 先算 fitness 再标退化覆盖 — 检查真实代码)
old = '''        df["fitness"] = (df["ICIR"]
                         - self.LAMBDA_TURNOVER * df["turnover"]
                         - self.MU_CORR * df["corr_penalty"])
        return df, S'''
new = '''        # 退化检测: 截面方差≈0 的因子无排序信息, fitness 压到永不进精英池
        row_std = torch.nan_to_num(S).std(dim=(1, 2))          # [N]
        deg = row_std < 1e-9
        df["degenerate"] = deg.cpu().numpy()
        df["fitness"] = (df["ICIR"]
                         - self.LAMBDA_TURNOVER * df["turnover"]
                         - self.MU_CORR * df["corr_penalty"])
        if deg.any():
            print(f"  [dedup] {int(deg.sum())}/{len(df)} 退化因子已过滤 (截面方差≈0)")
            df.loc[deg, "fitness"] = -1e9
        return df, S'''
assert old in s, 'fitness tail not found'
s = s.replace(old, new)

# 3) merge_elite_bucketed: 结构签名去重 (按实际缩进重写)
old_merge = '''        pool = {}
        for e in cand_trees + list(elite_by_name.values()):
            pool[expr_str(e)] = e
        buckets: dict[str, list] = defaultdict(list)
        for s, e in pool.items():
            buckets[bucket_of(e)].append((cand_fit.get(s, -9e9), s, e))
        merged: dict[str, tuple] = {}'''
new_merge = '''        pool = {}
        for e in cand_trees + list(elite_by_name.values()):
            pool[expr_str(e)] = e
        # 结构签名去重: 同签名只留 fitness 最高; DEGENERATE 直接排除
        sig_fit: dict[str, float] = {}
        sig_tree: dict[str, tuple] = {}
        for s_, e_ in pool.items():
            sig = structural_signature(e_)
            if sig == "DEGENERATE":
                continue
            f_ = cand_fit.get(s_, -9e9)
            if sig not in sig_fit or f_ > sig_fit[sig]:
                sig_fit[sig] = f_
                sig_tree[sig] = e_
        pool = {expr_str(e): e for e in sig_tree.values()}
        buckets: dict[str, list] = defaultdict(list)
        for s, e in pool.items():
            buckets[bucket_of(e)].append((cand_fit.get(s, -9e9), s, e))
        merged: dict[str, tuple] = {}'''
assert old_merge in s, 'merge head not found'
s = s.replace(old_merge, new_merge)

# 4) refine: 退化变体先剔再选最优
old_ref = '''        dfv["parent_id"] = owner
        best_idx = dfv.groupby("parent_id")["fitness"].idxmax()
        keep_pos = best_idx.to_numpy()'''
new_ref = '''        dfv["parent_id"] = owner
        dfv = dfv[~dfv["degenerate"].astype(bool)]
        best_idx = dfv.groupby("parent_id")["fitness"].idxmax()
        keep_pos = best_idx.to_numpy()'''
assert old_ref in s, 'refine tail not found'
s = s.replace(old_ref, new_ref)

p.write_text(s, encoding='utf-8')
ast.parse(s)
print('part2 OK — all four patches applied')
