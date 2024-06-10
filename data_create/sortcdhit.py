import matplotlib.pyplot as plt

# テキストファイルを読み込む
with open("/home/aca10223gf/workplace/data/OMA_database/oma-groups.cdhit_80.txt", "r") as f:
    lines = f.readlines()

# 行をグループIDでソート
sorted_lines = sorted(lines, key=lambda x: int(x.split("\t")[0]))

# # ソート後の行を新しいファイルに書き込む
# with open("/home/aca10223gf/workplace/data/OMA_database/oma-groups.cdhit_80_sorted.txt", "w") as f:
#     for line in sorted_lines:
#         f.write(line)

# # 配列IDの数をカウント（単独のIDは除外）
# seq_ids = [seq_id for line in sorted_lines for seq_id in line.split("\t")[1:] if len(line.split("\t")) > 2]
# print(f"Number of sequence IDs (excluding singleton IDs): {len(seq_ids)}")



# グループ内の配列数を数える

group_sizes = [len(line.split("\t")) - 1 for line in sorted_lines if len(line.split("\t")) > 2]
max_size = max(group_sizes)
# ヒストグラムを描画
plt.figure(figsize=(10, 6))
plt.hist(group_sizes, bins=50, range=(0, max_size), alpha=0.7, color='blue')
plt.xlabel('Number of sequences in a group')
plt.ylabel('Frequency')
plt.title('Histogram of group sizes')
plt.grid(True)

# 図を保存
plt.savefig("/home/aca10223gf/workplace/data/OMA_database/group_sizes_histogram.png")

# グループサイズごとの総配列数を計算
max_size = max(group_sizes)
size_counts = [0] * (max_size + 1)
for size in group_sizes:
    size_counts[size] += size

# 累積総配列数を計算
cumulative_counts = [sum(size_counts[i:]) for i in range(len(size_counts))]

# 積み上げ棒グラフを描画
plt.figure(figsize=(10, 6))
plt.bar(range(len(size_counts)), cumulative_counts, width=1.0, edgecolor='black', linewidth=0.5)
plt.xlabel('Number of sequences in a group')
plt.ylabel('Cumulative number of sequences')
plt.title('Cumulative distribution of sequences by group size')
plt.grid(True)

# 図を保存
plt.savefig("/home/aca10223gf/workplace/data/OMA_database/cumulative_distribution.png")