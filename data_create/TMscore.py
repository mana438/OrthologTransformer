# from tmscoring import get_tm
# tm_score = get_tm('/home/4/ux03574/workplace/sample/ortholog_10_predicted_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb', '/home/4/ux03574/workplace/sample/ortholog_24_target_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb')

# print(tm_score)
from Bio import SeqIO
import os
import glob
import matplotlib.pyplot as plt
from tmscoring import get_tm

# 入力ディレクトリ
base_dir = "/home/4/ux03574/workplace/data/PDB/BS_IS"
fasta_path = os.path.join(base_dir, "source_target_predicted_protein.fasta")

# FASTAのID順に取得（ortholog_1_source_protein など）
records = list(SeqIO.parse(fasta_path, "fasta"))
num_orthologs = len(records) // 3

x_scores = []  # TMscore: source vs target（横軸）
y_scores = []  # TMscore: source vs predicted（縦軸）
labels = []

for i in range(1, num_orthologs + 1):
    prefix = f"ortholog_{i}"
    source = os.path.join(base_dir, f"{prefix}_source_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb")
    target = os.path.join(base_dir, f"{prefix}_target_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb")
    predicted = os.path.join(base_dir, f"{prefix}_predicted_protein_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb")

    if not (os.path.exists(source) and os.path.exists(target) and os.path.exists(predicted)):
        print(f"[!] Skipping {prefix}: missing PDB file(s)")
        continue

    try:
        tm_source_target = get_tm(source, target)
        tm_source_pred = get_tm(source, predicted)

        x_scores.append(tm_source_target)
        y_scores.append(tm_source_pred)
        labels.append(prefix)

    except Exception as e:
        print(f"[!] Error computing TMscore for {prefix}: {e}")

# 保存先ディレクトリ
output_dir = "/home/4/ux03574/workplace/job_results/BS_IS"
os.makedirs(output_dir, exist_ok=True)

# プロット作成
plt.figure(figsize=(10, 10))
plt.scatter(x_scores, y_scores, color='blue', s=80)

# 軸・タイトル・装飾
plt.xlabel("TMscore: source vs target", fontsize=14)
plt.ylabel("TMscore: source vs predicted", fontsize=14)
plt.title("Structural similarity of orthologs (TMscore)", fontsize=16)
plt.grid(True)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # 対角線

# 自動で表示範囲を調整
x_margin = 0.05
y_margin = 0.05
x_min = max(0.0, min(x_scores) - x_margin)
x_max = min(1.0, max(x_scores) + x_margin)
y_min = max(0.0, min(y_scores) - y_margin)
y_max = min(1.0, max(y_scores) + y_margin)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 保存
plot_path = os.path.join(output_dir, "tmscore_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[+] Plot saved to: {plot_path}")

plt.show()