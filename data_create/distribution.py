from Bio import SeqIO
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import os  # ファイル名を扱うために os モジュールをインポート
from Bio.Data import CodonTable  # コドン表を取得するためにインポート

def get_codon_distribution(fasta_file):
    codon_counts = Counter()
    total_codons = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).upper()
        # 配列長を3の倍数に調整（末尾を切り捨て）
        seq = seq[:(len(seq)//3)*3]
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            if len(codon) == 3 and all(nuc in 'ATGC' for nuc in codon):
                codon_counts[codon] += 1
                total_codons += 1
            else:
                continue  # コドンに非ATGC文字が含まれる場合は無視

    # コドンの出現回数を返す
    return codon_counts

def calculate_cai(seq_records, ref_codon_freq):
    cai_values = []
    codon_table = CodonTable.unambiguous_dna_by_name["Standard"]

    # 各コドンの相対適応度（相対頻度）を計算
    codon_relative_adaptiveness = {}
    for aa in codon_table.forward_table.values():
        synonymous_codons = [codon for codon, amino_acid in codon_table.forward_table.items() if amino_acid == aa]
        max_freq = max([ref_codon_freq.get(codon, 0) for codon in synonymous_codons])
        for codon in synonymous_codons:
            freq = ref_codon_freq.get(codon, 0)
            if max_freq > 0:
                codon_relative_adaptiveness[codon] = freq / max_freq
            else:
                codon_relative_adaptiveness[codon] = 0

    # 各配列についてCAIを計算
    for record in seq_records:
        seq = str(record.seq).upper()
        # 配列長を3の倍数に調整（末尾を切り捨て）
        seq = seq[:(len(seq)//3)*3]
        codon_list = [seq[i:i+3] for i in range(0, len(seq), 3)]
        valid_codons = [codon for codon in codon_list if codon in codon_relative_adaptiveness]

        if not valid_codons:
            continue

        w_values = [codon_relative_adaptiveness[codon] for codon in valid_codons if codon_relative_adaptiveness[codon] > 0]
        if not w_values:
            continue

        # CAIの計算
        cai = np.exp(np.sum(np.log(w_values)) / len(w_values))
        cai_values.append(cai)

    # 平均CAIを返す
    if cai_values:
        return np.mean(cai_values)
    else:
        return None

# FASTAファイルのパスを指定
fasta_file1 = '/home/4/ux03574/workplace/data/AI_petase/FAST.fasta'
fasta_file2 = '/home/4/ux03574/workplace/data/OMA_database/BACSU.fasta'

# ファイル名（拡張子なし）を取得
file_name1 = os.path.splitext(os.path.basename(fasta_file1))[0]
file_name2 = os.path.splitext(os.path.basename(fasta_file2))[0]

# 全ての可能なコドンのリストを作成
codons = [a + b + c for a in 'ATGC' for b in 'ATGC' for c in 'ATGC']

# 2つのFASTAファイルからコドン分布（出現回数）を取得
codon_counts1 = get_codon_distribution(fasta_file1)
codon_counts2 = get_codon_distribution(fasta_file2)

# 標準コドン表を取得
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
codon_to_aa = standard_table.forward_table.copy()
# ストップコドンを '*' にマッピング
for codon in standard_table.stop_codons:
    codon_to_aa[codon] = '*'

# アミノ酸から同義コドンのリストを作成
aa_to_codons = {}
for codon, aa in codon_to_aa.items():
    aa_to_codons.setdefault(aa, []).append(codon)

# アミノ酸のリストを取得
amino_acids = list(aa_to_codons.keys())

# 各アミノ酸についてJensen-Shannon divergenceを計算
divergences = []
for aa in amino_acids:
    codons_for_aa = aa_to_codons[aa]
    # 各ファイルでの当該アミノ酸のコドン出現回数を取得
    counts1_aa = np.array([codon_counts1.get(codon, 0) for codon in codons_for_aa])
    counts2_aa = np.array([codon_counts2.get(codon, 0) for codon in codons_for_aa])
    # 出現回数の合計がゼロの場合はスキップ
    if counts1_aa.sum() == 0 or counts2_aa.sum() == 0:
        continue
    # 出現回数を頻度に正規化
    freqs1_aa = counts1_aa / counts1_aa.sum()
    freqs2_aa = counts2_aa / counts2_aa.sum()
    # ゼロを回避するために小さな値を加算
    epsilon = 1e-10
    freqs1_aa += epsilon
    freqs2_aa += epsilon
    # 再度正規化
    freqs1_aa /= freqs1_aa.sum()
    freqs2_aa /= freqs2_aa.sum()
    # Jensen-Shannon divergenceを計算
    jsd = jensenshannon(freqs1_aa, freqs2_aa, base=2)
    divergences.append(jsd)

# 平均のJensen-Shannon divergenceを計算
if divergences:
    avg_js_divergence = np.mean(divergences)
    print(f'Average Jensen-Shannon divergence: {avg_js_divergence}')
else:
    print('No amino acids with codon frequencies to compare.')

# 全体のコドン頻度を計算（グラフ作成のため）
total_counts1 = sum(codon_counts1.values())
total_counts2 = sum(codon_counts2.values())
codon_freq1 = {codon: codon_counts1.get(codon, 0) / total_counts1 for codon in codons}
codon_freq2 = {codon: codon_counts2.get(codon, 0) / total_counts2 for codon in codons}

# CAIの計算
seq_records1 = list(SeqIO.parse(fasta_file1, "fasta"))
ref_codon_freq = codon_freq2  # fasta_file2のコドン頻度を参照
cai_value = calculate_cai(seq_records1, ref_codon_freq)
if cai_value is not None:
    print(f'CAI value: {cai_value}')
else:
    print('CAI could not be calculated.')

# コドン頻度の配列を作成
freqs1 = np.array([codon_freq1.get(codon, 0) for codon in codons])
freqs2 = np.array([codon_freq2.get(codon, 0) for codon in codons])

# コドン頻度の比較をプロット
x = np.arange(len(codons))
width = 0.35

fig, ax = plt.subplots(figsize=(20, 10))
rects1 = ax.bar(x - width/2, freqs1, width, label=file_name1)
rects2 = ax.bar(x + width/2, freqs2, width, label=file_name2)

ax.set_xlabel('Codon')
ax.set_ylabel('Frequency')
ax.set_title('Comparison of Codon Frequencies')
ax.set_xticks(x)
ax.set_xticklabels(codons, rotation=90)
ax.legend()

# グラフ内に平均のJensen-Shannon divergenceとCAIの値を表示
text_str = f'Average Jensen-Shannon divergence: {avg_js_divergence:.4f}\nCAI: {cai_value:.4f}'
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14, verticalalignment='top')

plt.tight_layout()

# 画像のファイル名を作成
output_filename = f'/home/4/ux03574/workplace/job_results/{file_name1}_vs_{file_name2}_codon_frequency_comparison.png'
plt.savefig(output_filename)  # 画像を保存
plt.show()
