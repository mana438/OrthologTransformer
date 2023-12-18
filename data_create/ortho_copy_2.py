import os
import shutil
import random

# 指定のディレクトリ
current_dir = "/home/aca10223gf/workplace/data/CDS_aa/OrthoFinder/Results_Jul16/Orthologues"

# コピー先のディレクトリ
dest_dir = "/home/aca10223gf/workplace/data/sample_ortho/sample_train_3"

# 総行数目標
total_lines_target = 300000

# 現在のディレクトリ下のすべてのディレクトリを取得
all_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

# 全ファイルの行数を取得
total_lines_all = 0
for directory in all_dirs:
    all_files = os.listdir(os.path.join(current_dir, directory))
    for file in all_files:
        file_path = os.path.join(current_dir, directory, file)
        with open(file_path, 'r') as f:
            total_lines_all += sum(1 for _ in f)

# 間引く確率を計算
prob = total_lines_target / total_lines_all

for directory in all_dirs:
    # 各ディレクトリ内の全ファイルを取得
    all_files = os.listdir(os.path.join(current_dir, directory))
    all_files.sort()  # ファイルを順番に処理するためソート

    for file in all_files:
        file_path = os.path.join(current_dir, directory, file)

        # 新しいディレクトリパスを作成
        new_dir_path = os.path.join(dest_dir, directory)
        os.makedirs(new_dir_path, exist_ok=True)

        new_file_path = os.path.join(new_dir_path, file)

        with open(file_path, 'r') as f_in, open(new_file_path, 'w') as f_out:
            lines = f_in.readlines()
            # ヘッダー行は必ず含める
            f_out.write(lines[0])
            # それ以降の行は確率的に選ぶ
            for line in lines[1:]:
                if random.random() < prob:
                    f_out.write(line)
