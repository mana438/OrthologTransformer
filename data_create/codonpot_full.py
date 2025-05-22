#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# 目的:
# ・test_fasta ディレクトリにあるファイル群を読み込み、2 本目 (source) の配列を
#   1 本目 (target) の菌種に合わせてコドン最適化する。
# ・コドン使用頻度は、train_fasta ディレクトリに保存されている同じ菌種
#   (ヘッダー先頭 5 文字) の全配列を用いて決定する。
# ・最終的に下記 3 種類の配列を 1 つの FASTA にまとめ、
#   job_results/BS_IS/ に保存する。
#     1) source_sequence   … 各テストファイル 2 本目
#     2) target_sequence   … 各テストファイル 1 本目
#     3) codonopt_sequence … 2) の菌種向けに最適化した 1) の配列
# ・さらに util.alignment を用いてアラインメントスコアのプロットを作成し、
#   同フォルダーに保存する。
###############################################################################

from pathlib import Path
from collections import Counter, defaultdict
from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
import itertools
from util import save_with_unique_filename
from Bio import pairwise2
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["BS_IS", "BS_SE", "E1_MT", "E1_PP", "E1_SJ",
            "LA_AV", "LL_TT", "PF_RL", "ST_DR"]
            
def process_dataset(tag: str):
    # ---------- データセット固有パス ----------
    test_dir   = Path(f"/home/4/ux03574/workplace/data/OMA_database/{tag}/test_fasta/")
    train_dir  = Path(f"/home/4/ux03574/workplace/data/OMA_database/{tag}/train_fasta/")
    result_dir = Path(f"/home/4/ux03574/workplace/job_results/{tag}/")
    result_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- テストファイルのヘッダーから菌種名取得 -------------- #
    # すべてのファイルで 1 本目の菌種は共通 (例: 'BACSUxxxxx')
    # 最初に 1 ファイル読んで菌種コード (5 文字) を抽出する
    sample_record = next(SeqIO.parse(next(test_dir.glob("*.fasta")), "fasta"))
    target_species = sample_record.id[:5]              # 'BACSU'
    
    print(f"[INFO] コドン最適化対象の菌種コード: {target_species}")
    
    # ---------------------- トレーニング配列からコドン使用頻度算出 --------------- #
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"].forward_table
    # ※ STOP コドンは forward_table には含まれないので無視して良い
    
    # 各アミノ酸 → Counter(コドン→出現数)
    codon_usage: dict[str, Counter[str]] = defaultdict(Counter)
    
    for fasta_path in train_dir.glob("*.fasta"):
        for record in SeqIO.parse(fasta_path, "fasta"):
            if record.id.startswith(target_species):
                seq_str = str(record.seq).upper()
                # 3 塩基ずつ走査（途中で割り切れない場合は末端を捨てる）
                for i in range(0, len(seq_str) - len(seq_str) % 3, 3):
                    codon = seq_str[i : i + 3]
                    # N 含むコドン等の例外処理
                    if codon in standard_table:
                        aa = standard_table[codon]
                        codon_usage[aa][codon] += 1
    
    # 各アミノ酸で最頻コドンを決定
    major_codon: dict[str, str] = {
        aa: counter.most_common(1)[0][0] for aa, counter in codon_usage.items()
    }
    
    print("[INFO] 最頻コドン表を構築しました。例:")
    for aa, cod in itertools.islice(major_codon.items(), 10):
        print(f"  {aa}: {cod}")
    
    # ---------------------- テスト配列を読み込み & コドン最適化 ------------------- #
    source_sequences      = []   # 元々 2 本目の配列
    target_sequences      = []   # 1 本目の配列
    codonopt_sequences    = []   # 新しく作成する配列
    fasta_entries_output  = []   # FASTA 書き出し用 SeqRecord リスト
    
    for fasta_path in sorted(test_dir.glob("*.fasta")):
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if len(records) != 2:
            raise ValueError(f"{fasta_path} は 2 本の配列を持っていません。")
        
        target_rec, source_rec = records  # 指定通り 1 本目が target, 2 本目が source
        
        # ---------- コドン最適化: アミノ酸配列を保ちつつコドン置換 ----------
        optimized_codons = []
        src_seq_str = str(source_rec.seq).upper()
        
        for i in range(0, len(src_seq_str) - len(src_seq_str) % 3, 3):
            codon = src_seq_str[i : i + 3]
            aa = standard_table.get(codon)  # get で None 回避 (Stop/不正コドンは None)
            if aa and aa in major_codon:
                optimized_codons.append(major_codon[aa])
            else:
                optimized_codons.append(codon)  # 変換できない場合はそのまま
        
        # 余った塩基 (長さ % 3) は末尾にそのまま付加
        remainder = src_seq_str[len(src_seq_str) - len(src_seq_str) % 3 :]
        optimized_seq_str = "".join(optimized_codons) + remainder
        
        # ---------- リストに格納 ----------
        source_sequences.append(src_seq_str)
        target_sequences.append(str(target_rec.seq).upper())
        codonopt_sequences.append(optimized_seq_str)
        
        # ---------- FASTA 書き出し用 SeqRecord ----------
        fasta_entries_output.extend([
            SeqIO.SeqRecord(Seq(src_seq_str),      id=f"{fasta_path.stem}|source", description=""),
            SeqIO.SeqRecord(Seq(str(target_rec.seq).upper()), id=f"{fasta_path.stem}|target", description=""),
            SeqIO.SeqRecord(Seq(optimized_seq_str), id=f"{fasta_path.stem}|codonopt", description="")
        ])
    
    # ---------------------------- FASTA を保存 ---------------------------------- #
    output_fasta = result_dir / "codon_optimization_results.fasta"
    SeqIO.write(fasta_entries_output, output_fasta, "fasta")
    print(f"[INFO] すべての配列を書き出しました: {output_fasta}")
    
    
    ###############################################################################
    # ★★★ ここからアラインメント関連の処理を全面的に書き換えました ★★★
    ###############################################################################
    # ・コドン(3 塩基)単位のグローバルアラインメントを行い、スコアを算出する
    #     ─ マッチ     : +1
    #     ─ ミスマッチ :  0
    #     ─ ギャップ   :  0  (gap_open = gap_extend = 0)
    # ・算出したスコアを「ギャップを含むアラインメント配列長」で割り正規化
    # ・source‑vs‑target (x 軸) と codonopt‑vs‑target (y 軸) の散布図を作成
    ###############################################################################
    
    
    # ------------------- コドン⇔1 文字へのマッピングを用意 --------------------- #
    # 64 通りのコドンを Base64 文字列 (A‑Z a‑z 0‑9 + /) に対応づける
    char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    # 標準テーブルに載っている 61 コドン (+ Stop を含め計 64 個確保)
    all_codons = sorted(set(standard_table.keys()) | {"TAA", "TAG", "TGA"})
    codon2char = {c: char_set[i] for i, c in enumerate(all_codons)}
    unknown_char = "?"  # N を含むなど未知コドン用
    
    def codon_seq_to_char_string(seq: str) -> str:
        """3 塩基ずつ区切って 1 文字列に変換 (末端の余りは捨てる)"""
        chars = []
        for i in range(0, len(seq) - len(seq) % 3, 3):
            codon = seq[i : i + 3]
            chars.append(codon2char.get(codon, unknown_char))
        return "".join(chars)
    
    def calc_codon_alignment_ratio(seqA: str, seqB: str) -> float:
        """コドン単位グローバルアラインメント比率を返す"""
        s1 = codon_seq_to_char_string(seqA)
        s2 = codon_seq_to_char_string(seqB)
    
        # pairwise2 の globalxx: match=1, mismatch=0, gap_open=gap_extend=0
        aln = pairwise2.align.globalxx(s1, s2, one_alignment_only=True, score_only=False)[0]
        score = aln.score                       # マッチ数
        tgt_length = len(s2)                    # ★ 変更: 正規化は target (s2) の長さ
    
        return score / tgt_length if tgt_length else 0.0   # ★ 変更: 分母を tgt_length に
        
    # --------------------- スコア計算 & 散布図用データ集計 ---------------------- #
    src_vs_tgt_scores  = []
    opt_vs_tgt_scores  = []
    
    for src, tgt, opt in zip(source_sequences, target_sequences, codonopt_sequences):
        src_vs_tgt_scores.append(calc_codon_alignment_ratio(src, tgt))
        opt_vs_tgt_scores.append(calc_codon_alignment_ratio(opt, tgt))
    
    mean_src = np.mean(src_vs_tgt_scores)
    mean_opt = np.mean(opt_vs_tgt_scores)
    
    # ---------------------------- 散布図の作成 --------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(src_vs_tgt_scores, opt_vs_tgt_scores, alpha=0.8)
    
    # 平均位置の補助線
    ax.axvline(mean_src, color="black", linestyle=":", linewidth=1, zorder=0)
    ax.axhline(mean_opt, color="black", linestyle=":", linewidth=1, zorder=0)
    
    # ★ 軸座標系 (0–1 範囲) に対して少し外側へラベルを配置
    # ★★★ ここだけ変更してください（平均値ラベルの位置を軸内・左上に移動） ★★★
    # ── 既存の ax.text(...) 2 行を削除して、下記に差し替え ──
    ax.text(0.02, 0.95, f"μₓ = {mean_src:.3f}",
            transform=ax.transAxes,        # 両方とも軸座標 (0‑1) で指定
            ha="left", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none"))
    
    ax.text(0.02, 0.88, f"μ_y = {mean_opt:.3f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none"))
    
    
    # y = x の目安線
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    
    ax.set_xlabel("Source ↔ Target  codon‑alignment score")
    ax.set_ylabel("Optimized ↔ Target  codon‑alignment score")
    ax.set_title("Codon‑level alignment score comparison")
    
    # ★ 軸を少し拡張してラベルのための余白を確保
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    # --------------------------- プロットの保存 -------------------------------- #
    output_png = result_dir / "codonopt_alignment_scatter.png"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")  # ★ 通常の savefig で保存
    print(f"[INFO] 散布図を保存しました: {output_png}")
    
if __name__ == "__main__":
    for tag in DATASETS:
        print(f"\n======== {tag} 開始 ========")
        process_dataset(tag)
