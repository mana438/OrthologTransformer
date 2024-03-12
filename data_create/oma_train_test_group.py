def read_and_process_file(input_filename):
    # ファイルを読み込む
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # 各グループのデータを格納するリスト
    large_group = []
    small_group = []
    current_group = None

    # ファイルの内容を解析
    for line in lines:
        line = line.strip()  # 余分な空白を削除
        if line == "Large Group:":
            current_group = large_group
        elif line == "Small Group:":
            current_group = small_group
        elif line.startswith("OMAGrp"):
            # OMAGrpを除去し、整形して追加
            number_part = line.replace("OMAGrp", "")
            # 先頭のゼロを除去し、整形した番号を追加
            formatted_number = str(int(number_part))
            current_group.append(formatted_number)

    # 各グループを別のファイルに保存
    save_to_file('/home/aca10223gf/workplace/data/OMA_database/train_group.txt', large_group)
    save_to_file('/home/aca10223gf/workplace/data/OMA_database/test_group.txt', small_group)

def save_to_file(filename, data):
    # データをファイルに保存
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

# 元のファイル名を指定して処理を実行
read_and_process_file('/home/aca10223gf/workplace/data/OMA_database/group_division.txt')
