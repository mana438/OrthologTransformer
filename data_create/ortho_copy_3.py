import os
import glob
from collections import defaultdict

# Orthofinderファイルを読み込む関数
def read_orthofinder_files(directory):
    files = glob.glob(os.path.join(directory, '*.txt'))  # 必要に応じて拡張子を変更
    gene_sets_per_file = []
    headers = []
    total_lines = 0

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            headers.append(lines[0].strip())
            # 各行をそのままの形式でリストに格納
            gene_sets_per_file.append(lines[1:])
            total_lines += len(lines) - 1

    return gene_sets_per_file, headers, files, total_lines

# 各行に含まれる全ての遺伝子を抽出する関数
def extract_genes(line):
    # タブとカンマで分割して全ての遺伝子を抽出
    genes = line.strip().split('\t')
    return [gene for part in genes for gene in part.split(',')]

# 各ファイルから比例的に行を選択する関数
def select_proportional_rows(gene_sets_per_file, total_lines):
    unique_genes = set()
    for gene_sets in gene_sets_per_file:
        for line in gene_sets:
            genes = extract_genes(line)
            unique_genes.update(genes)

    selected_rows = []  # 選択された行（元の形式）を格納するリスト
    used_genes = set()
    lines_per_file = [len(gene_sets) for gene_sets in gene_sets_per_file]
    proportional_limits = [int((lines / total_lines) * 2 * len(unique_genes)) for lines in lines_per_file]

    for file_index, gene_sets in enumerate(gene_sets_per_file):
        limit = proportional_limits[file_index]
        for line in gene_sets:
            genes = extract_genes(line)
            if len(selected_rows) < limit and not used_genes.issuperset(genes):
                selected_rows.append((file_index, line))
                used_genes.update(genes)

    return selected_rows

# 選択された行を新しいファイルに書き出す関数
def write_selected_rows(selected_rows, headers, files, output_dir):
    rows_per_file = defaultdict(list)
    for i, line in selected_rows:
        rows_per_file[i].append(line)
    
    for i, file in enumerate(files):
        with open(os.path.join(output_dir, os.path.basename(file)), 'w') as f:
            f.write(headers[i] + '\n')
            for row in rows_per_file[i]:
                f.write(row)

# 最終的な統計情報を出力する関数
def print_stats(selected_rows, unique_genes, used_genes, lines_per_file):
    print("統計情報:")
    print(f"全ユニーク遺伝子数: {len(unique_genes)}")
    print(f"使用された遺伝子数: {len(used_genes)}")
    print("全遺伝子が使用されたかどうか:", "はい" if used_genes == unique_genes else "いいえ")
    file_rows = defaultdict(int)
    for i, _ in selected_rows:
        file_rows[i] += 1
    for i, count in file_rows.items():
        print(f"ファイル {i + 1} から選択された行数: {count} / {lines_per_file[i]}")

# メイン関数
def main():
    directory = 'path/to/your/directory'
    output_dir = 'path/to/output/directory'
    gene_sets_per_file, headers, files, total_lines = read_orthofinder_files(directory)
    selected_rows = select_proportional_rows(gene_sets_per_file, total_lines)
    write_selected_rows(selected_rows, headers, files, output_dir)

    # 統計情報の出力
    unique_genes = set()
    used_genes = set()
    for gene_sets in gene_sets_per_file:
        for line in gene_sets:
            genes = extract_genes(line)
            unique_genes.update(genes)
    for _, line in selected_rows:
        genes = extract_genes(line)
        used_genes.update(genes)
    print_stats(selected_rows, unique_genes, used_genes, [len(gene_sets) for gene_sets in gene_sets_per_file])

if __name__ == "__main__":
    main()