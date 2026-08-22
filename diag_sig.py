import ast
import pathlib

p = pathlib.Path('/mnt/d/quant/scripts/factor_mining/search.py')
s = p.read_text(encoding='utf-8')

# 剩余近亲是窗口变体: ts_min(ts_delta(cs_zscore(macd_hist) 5) 120) vs ...60)
# 它们来自 refine (窗口重映射) — 全局签名去重段用的正则把窗口数字替换成 <W>,
# 但 ts_delta 的参数在 DELTAS=(1,5,10,20), 5 是 delta 不是 window — 归一化
# 应该没问题... 真正原因: 该去重段只在 ref_df 分支执行, 但 df 拼接顺序是
# [原精英, ref 变体], 排序按 ICIR 后 keep_rows 遍历 — 逻辑对.
# 但看输出, cs_zscore(macd_hist) 外壳透明化后 ts_delta(cs_zscore(x) 5) 与
# ts_delta(x 5) 同签名 → 已合并. 剩下的差异是真窗口差异 (120/60/20/5).
# 这些是"真变体": 窗口不同输出不同. 但同 ICIR +0.0124 说明输出几乎相同
# (ts_min 对窗口不敏感). 这是窗口近亲问题 — 签名设计上窗口归一为 <W>
# 只在 structural_signature 里做, 而全局去重段用正则近似. 检查正则是否
# 匹配到 " 60)" / " 120)" — (?<![\w)])\b(5|10|20|60|120)\b(?![\w)])
# "ts_delta(cs_zscore(macd_hist) 5)" 中 " 5)" 的 5 前面是空格, 后面 ')' 
# → 应匹配. 但 "macd_hist" 含 _hist? \w 包含下划线, hist 的 t 前...
# " 120) 60)" 两处都该被替换. 让我实测签名.

# 直接验证: 打印这些行的实际 factor 字符串和它们的归一化签名
import re, sys
sys.path.insert(0, 'scripts')
factors = [
    'ts_max(ts_min(ts_delta(cs_zscore(macd_hist) 5) 120) 120)',
    'ts_mean(ts_min(ts_delta(cs_zscore(macd_hist) 5) 120) 120)',
    'ts_max(ts_min(ts_delta(cs_zscore(macd_hist) 5) 120) 60)',
]
for f in factors:
    sig = re.sub(r"-?\d+\.\d+", "<C>", f)
    sig = re.sub(r"(cs_rank|cs_zscore|cs_demean)\(", "", sig)
    sig = re.sub(r"(?<![\w)])\b(5|10|20|60|120)\b(?![\w)])", "<W>", sig)
    print(sig)
PYEOF_MARKER_NOT_USED = None
