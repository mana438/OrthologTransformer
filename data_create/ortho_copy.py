import os
import shutil

# 指定のディレクトリ
current_dir = "/home/aca10223gf/workplace/data/CDS_aa/OrthoFinder/Results_Apr26/Orthologues"

# コピー先のディレクトリ
dest_dir = "/home/aca10223gf/workplace/data/sample_ortho/sample_train"

# 現在のディレクトリ下のすべてのディレクトリを取得
all_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

for directory in all_dirs:
    # 各ディレクトリ内の全ファイルを取得
    all_files = os.listdir(os.path.join(current_dir, directory))
    
    # 条件に適合するファイルを選ぶための初期化
    largest_file = None
    largest_size = 0

    for file in all_files:
        file_path = os.path.join(current_dir, directory, file)
        # ファイルの行数を取得
        num_lines = sum(1 for line in open(file_path))
        # ファイルのサイズを取得
        file_size = os.path.getsize(file_path)

        # 条件を満たすかチェック
        if num_lines <= 4000 and file_size > largest_size:
            largest_file = file
            largest_size = file_size

    # 条件を満たす最大のファイルをコピー
    if largest_file is not None:
        shutil.copy2(os.path.join(current_dir, directory, largest_file), os.path.join(dest_dir, largest_file))
