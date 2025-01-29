import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Data import CodonTable
from collections import defaultdict
import numpy as np
from Bio import SeqIO
import os
import glob

def read_sequence_pairs(folder_path):
   """
   指定フォルダ内の全FASTAファイルからtargetとinput配列を読み込む
   Returns:
       tuple: (target_sequences, input_sequences)
   """
   target_seqs = []
   input_seqs = []
   
   for fasta_file in glob.glob(os.path.join(folder_path, "*.fasta")):
       sequences = list(SeqIO.parse(fasta_file, "fasta"))
       if len(sequences) >= 2:
           target_seqs.append(str(sequences[0].seq))
           input_seqs.append(str(sequences[1].seq))
   
   return target_seqs, input_seqs

def calculate_gc_content(sequence):
   """GC含量を計算"""
   gc_count = sequence.count('G') + sequence.count('C')
   total = len(sequence)
   return gc_count / total * 100 if total > 0 else 0

def calculate_codon_frequencies(sequences):
   """コドン使用頻度を計算"""
   codon_count = defaultdict(int)
   total_codons = 0
   
   for seq in sequences:
       for i in range(0, len(seq), 3):
           codon = seq[i:i+3]
           if len(codon) == 3:
               codon_count[codon] += 1
               total_codons += 1
   
   return {codon: count/total_codons for codon, count in codon_count.items()}

def calculate_relative_adaptiveness(codon_freq, codon_table):
    """相対適応度を計算"""
    w = {}
    # アミノ酸ごとのコドンをグループ化
    aa_to_codons = defaultdict(list)
    for codon, aa in codon_table.forward_table.items():
        if isinstance(codon, str):  # 単一コドンの場合
            aa_to_codons[aa].append(codon)
            
    # print(aa_to_codons)
    
    # 各アミノ酸グループ内で相対適応度を計算
    for aa, codons in aa_to_codons.items():
        frequencies = [codon_freq.get(codon, 0) for codon in codons]
        max_freq = max(frequencies)
        if max_freq > 0:
            for codon in codons:
                w[codon] = codon_freq.get(codon, 0) / max_freq
    
    # print("Number of codons with non-zero w:", sum(1 for v in w.values() if v > 0))
    # print("Sample of w values:", dict(list(w.items())[:5]))
    return w

def calculate_cai(sequence, w):
    """CAIを計算"""
    if len(sequence) < 3:
        return 0
    
    cai_values = []
    # シーケンスをコドンに分割
    codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
    
    # 各コドンのwの対数を計算
    for codon in codons:
        if len(codon) == 3 and codon in w and w[codon] > 0:
            cai_values.append(np.log(w[codon]))
    
    if not cai_values:
        return 0
        
    cai = np.exp(np.mean(cai_values))
    if cai > 1:  # これもエラーチェック
        print(f"Warning: CAI > 1: {cai}")
        return 1.0
    
    return cai

def plot_distributions(input_folder, base_file, finetuned_file, output_dir):
    """CAIとGC含量を縦に並べたバイオリンプロットを作成して保存"""
    # 配列の読み込み
    genomic_target_seqs, genomic_input_seqs = read_sequence_pairs(input_folder)
    base_seqs = [str(record.seq) for record in SeqIO.parse(base_file, "fasta")]
    finetuned_seqs = [str(record.seq) for record in SeqIO.parse(finetuned_file, "fasta")]

    # コドン使用頻度と相対適応度の計算
    codon_freq = calculate_codon_frequencies(genomic_target_seqs)
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
    w = calculate_relative_adaptiveness(codon_freq, standard_table)

    # ---- ① ループを使わず、CAI用データとGC用データを個別に作成 ----
    
    # CAIデータフレーム
    cai_data = {
        'Group': (
            ['Genomic input'] * len(genomic_input_seqs)
            + ['Genomic target'] * len(genomic_target_seqs)
            + ['Base'] * len(base_seqs)
            + ['Fine-tuned'] * len(finetuned_seqs)
        ),
        'Codon Adaptation Index': (
            [calculate_cai(seq, w) for seq in genomic_input_seqs]
            + [calculate_cai(seq, w) for seq in genomic_target_seqs]
            + [calculate_cai(seq, w) for seq in base_seqs]
            + [calculate_cai(seq, w) for seq in finetuned_seqs]
        )
    }
    df_cai = pd.DataFrame(cai_data)

    # GCデータフレーム
    gc_data = {
        'Group': (
            ['Genomic input'] * len(genomic_input_seqs)
            + ['Genomic target'] * len(genomic_target_seqs)
            + ['Base'] * len(base_seqs)
            + ['Fine-tuned'] * len(finetuned_seqs)
        ),
        'GC ratio': (
            [calculate_gc_content(seq) for seq in genomic_input_seqs]
            + [calculate_gc_content(seq) for seq in genomic_target_seqs]
            + [calculate_gc_content(seq) for seq in base_seqs]
            + [calculate_gc_content(seq) for seq in finetuned_seqs]
        )
    }
    df_gc = pd.DataFrame(gc_data)

    # ---- ② subplots を使って縦に2つのグラフを配置する ----
    # フォントサイズや描画領域などは好みに合わせて調整
    plt.rcParams['font.size'] = 30
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))

    # カラーパレットの設定
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # 青、緑、オレンジ、赤

    # ---- ③ それぞれのサブプロットに対してバイオリンプロットを作成 ----

    # 上段: CAI
    sns.violinplot(
        data=df_cai,
        x='Group',
        y='Codon Adaptation Index',
        hue='Group',
        palette=colors,
        ax=axes[0],
        legend=False
    )
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelbottom=False)  # 上段の目盛りラベルを表示しない
    axes[0].set_ylabel('Codon Adaptation Index', fontsize=20, labelpad=10)
    axes[0].tick_params(axis='y', labelsize=20)
    
    # 下段: GC ratio
    sns.violinplot(
        data=df_gc,
        x='Group',
        y='GC ratio',
        hue='Group',
        palette=colors,
        ax=axes[1],
        legend=False
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel('GC ratio', fontsize=20, labelpad=10)
    axes[1].tick_params(axis='x', rotation=45, labelsize=20)
    axes[1].tick_params(axis='y', labelsize=20)

    # レイアウト調整
    plt.tight_layout()

    # ---- ④ 画像を保存して閉じる ----
    output_path = os.path.join(output_dir, 'cai_gc_violin_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# 例: 変数を指定
# project_name = "BS_IS"  # ここを任意の文字列に変えて使う
# project_name = "BS_SE"
# project_name = "E1_MT"
# project_name = "E1_PP"
# project_name = "E1_SJ"
project_name = "LA_AV"
# project_name = "LL_TT"
# project_name = "PF_RL"
# project_name = "ST_DR"

# パス指定部分で f 文字列を使う
input_folder = f"/home/4/ux03574/workplace/data/OMA_database/{project_name}/test_fasta"
output_dir = f"/home/4/ux03574/workplace/job_results/{project_name}"

# plot_distributions 関数呼び出しの部分
plot_distributions(
    input_folder,
    f"/home/4/ux03574/workplace/job_results/{project_name}/predicted_sequences.fasta",
    f"/home/4/ux03574/workplace/job_results/{project_name}/predicted_sequences_1.fasta",
    output_dir
)

