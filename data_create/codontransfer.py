#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# CodonTransformer 版：source DNA → protein → CodonTransformer で DNA 逆翻訳
# 5 データセット（BS_IS, BS_SE, E1_MT, E1_PP, E1_SJ）について
#   ・source↔target コドン一致率
#   ・CodonTransformer ↔ target コドン一致率
# を算出し、散布図を保存する。
###############################################################################

from pathlib import Path
from collections import defaultdict, Counter
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import numpy as np

import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence

# ----------------------------- 設定 ---------------------------------------- #
DATASETS = ["BS_IS", "BS_SE", "E1_MT", "E1_PP", "E1_SJ"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
model     = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(device)

# Bacillus vs. E. coli の対応表
def organism_for_tag(tag: str) -> str:
    return "Bacillus subtilis" if tag.startswith("BS_") else "Escherichia coli general"

# ------------------- コドン→1 文字マッピング (64 種) ----------------------- #
char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
standard_table = CodonTable.unambiguous_dna_by_name["Standard"].forward_table
all_codons = sorted(set(standard_table.keys()) | {"TAA", "TAG", "TGA"})
codon2char = {c: char_set[i] for i, c in enumerate(all_codons)}
unknown_char = "?"

def codon_seq_to_char_string(seq: str) -> str:
    chars = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        codon = seq[i : i + 3]
        chars.append(codon2char.get(codon, unknown_char))
    return "".join(chars)

def calc_codon_alignment_ratio(seqA: str, seqB: str) -> float:
    s1 = codon_seq_to_char_string(seqA)
    s2 = codon_seq_to_char_string(seqB)
    aln = pairwise2.align.globalxx(s1, s2, one_alignment_only=True, score_only=False)[0]
    return aln.score / len(s2) if len(s2) else 0.0

# --------------------------- メイン処理 ------------------------------------ #
def process_dataset(tag: str):
    test_dir   = Path(f"/home/4/ux03574/workplace/data/OMA_database/{tag}/test_fasta/")
    result_dir = Path(f"/home/4/ux03574/workplace/job_results/{tag}/")
    result_dir.mkdir(parents=True, exist_ok=True)

    organism = organism_for_tag(tag)
    print(f"[{tag}] organism = {organism}")

    source_sequences  = []
    target_sequences  = []
    ct_sequences      = []

    fasta_out_records = []

    for fpath in sorted(test_dir.glob("*.fasta")):
        recs = list(SeqIO.parse(fpath, "fasta"))
        if len(recs) != 2:
            continue
        tgt_rec, src_rec = recs
        tgt_seq = str(tgt_rec.seq).upper()
        src_seq = str(src_rec.seq).upper()

        # ----- protein へ翻訳（末端 * を除去） -----
        protein = str(Seq(src_seq).translate(to_stop=False)).rstrip("*").replace("*", "")
        # ----- CodonTransformer で DNA 予測 -----
        ct_out = predict_dna_sequence(
            protein=protein,
            organism=organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=True
        )
        ct_seq = ct_out.predicted_dna.upper()

        # ----- リストに格納 -----
        source_sequences.append(src_seq)
        target_sequences.append(tgt_seq)
        ct_sequences.append(ct_seq)

        # ----- FASTA 出力用 -----
        stem = fpath.stem
        fasta_out_records.extend([
            SeqIO.SeqRecord(Seq(src_seq), id=f"{stem}|source", description=""),
            SeqIO.SeqRecord(Seq(tgt_seq), id=f"{stem}|target", description=""),
            SeqIO.SeqRecord(Seq(ct_seq),  id=f"{stem}|codonT", description="")
        ])

    # ---------- FASTA 書き出し ----------
    fasta_path = result_dir / "codontransformer_results.fasta"
    SeqIO.write(fasta_out_records, fasta_path, "fasta")
    print(f"[{tag}] FASTA 保存: {fasta_path}")

    # ---------- スコア計算 ----------
    src_scores = [calc_codon_alignment_ratio(s, t) for s, t in zip(source_sequences, target_sequences)]
    ct_scores  = [calc_codon_alignment_ratio(c, t) for c, t in zip(ct_sequences, target_sequences)]

    mean_src = np.mean(src_scores)
    mean_ct  = np.mean(ct_scores)

    # ---------- 散布図 ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(src_scores, ct_scores, alpha=0.8)
    ax.plot([0, 1], [0, 1], ls="--", lw=1)

    ax.axvline(mean_src, color="black", ls=":", lw=1, zorder=0)
    ax.axhline(mean_ct,  color="black", ls=":", lw=1, zorder=0)

    ax.text(0.02, 0.95, f"μₓ = {mean_src:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none"))
    ax.text(0.02, 0.88, f"μ_y = {mean_ct:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="none"))

    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Source ↔ Target  codon‑alignment score")
    ax.set_ylabel("CodonTransformer ↔ Target  codon‑alignment score")
    ax.set_title("Codon‑level alignment score comparison")

    png_path = result_dir / "codontransformer_alignment_scatter.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{tag}] 散布図保存: {png_path}")

# --------------------------- 実行 ------------------------------------------ #
if __name__ == "__main__":
    for tag in DATASETS:
        print(f"\n====== {tag} ======")
        process_dataset(tag)
