#!/usr/bin/env python3
"""
job_results/*_IDESA/ 内の codon_means.txt / aa_means.txt から
mean 値を集計して 1 枚の CSV にまとめるスクリプト
"""

import os
import re
import math
import pandas as pd

# -------------------------------------------------------------
# 1) job_results のルートディレクトリを指定
#    必ず自分の環境に合わせて変更してください
# -------------------------------------------------------------
root = "/home/4/ux03574/workplace/job_results"   # 例

# -------------------------------------------------------------
codon_re = re.compile(r"src\-tgt mean:\s*([0-9.+-eE]+).*?"
                      r"tgt\-pred mean:\s*([0-9.+-eE]+)", re.S)

rows = []
for entry in os.scandir(root):
    if entry.is_dir() and entry.name.endswith("_IDESA"):
        pair = entry.name                      # 例: ECO1A_IDESA
        c_src = c_pred = a_src = a_pred = math.nan

        # --- codon_means.txt ---
        c_path = os.path.join(entry.path, "codon_means.txt")
        if os.path.exists(c_path):
            with open(c_path) as fh:
                m = codon_re.search(fh.read())
                if m:
                    c_src, c_pred = map(float, m.groups())

        # --- aa_means.txt ---
        a_path = os.path.join(entry.path, "aa_means.txt")
        if os.path.exists(a_path):
            with open(a_path) as fh:
                m = codon_re.search(fh.read())
                if m:
                    a_src, a_pred = map(float, m.groups())

        rows.append({
            "pair": pair,
            "codon_src_tgt_mean": c_src,
            "codon_tgt_pred_mean": c_pred,
            "aa_src_tgt_mean": a_src,
            "aa_tgt_pred_mean": a_pred,
        })

# -------------------------------------------------------------
# 2) DataFrame → CSV 保存
# -------------------------------------------------------------
df = pd.DataFrame(rows).sort_values("pair")
csv_path = os.path.join(root, "summary_means.csv")
df.to_csv(csv_path, index=False)

print(f"✅ 集計完了: {csv_path}")
