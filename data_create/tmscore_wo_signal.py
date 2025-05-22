#!/usr/bin/env python3
"""
TM-score を一括計算するスクリプト
使用例:
    python calc_tmscores.py
"""

import os
from tmscoring import get_tm  # pip install tmscoring

# === 設定 ===
DIR_PATH = "/home/4/ux03574/workplace/data/PDB/BS_IS/PDB_WO_signal"
REF_PDB  = "fold_wt_model_0.pdb"
# ============

def main():
    ref_path = os.path.join(DIR_PATH, REF_PDB)

    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"基準 PDB が見つかりません: {ref_path}")

    results = []
    for fname in os.listdir(DIR_PATH):
        # PDB 以外や基準構造自身はスキップ
        if not fname.endswith(".pdb") or fname == REF_PDB:
            continue

        target_path = os.path.join(DIR_PATH, fname)
        # get_tm は (source, target) の順
        tm_score = get_tm(ref_path, target_path)
        results.append((fname, tm_score))

    # スコアが高い順に並べ替えて表示
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"=== TM-score (reference: {REF_PDB}) ===")
    for fname, score in results:
        print(f"{REF_PDB}  vs  {fname:<30} :  TM-score = {score:.5f}")

if __name__ == "__main__":
    main()
