from Bio import SeqIO
import random

# 菌種リストを読み込む関数
def load_species_list(species_list_file):
    with open(species_list_file, 'r') as file:
        # ファイルを読み込み、各行をセットに格納して返す
        species_list = {line.strip() for line in file}
    return species_list

def load_selected_groups(orthologous_groups_file, selected_groups_file, species_list):
    """
    指定されたグループに対応するオルソロググループと配列IDを辞書に登録する関数。
    
    Args:
    orthologous_groups_file (str): オルソロググループと配列IDが格納されたファイルのパス。
    selected_groups_file (str): 処理対象のグループ番号が記載されたファイルのパス。
    species_list (set): 処理対象の菌種リストが格納されたセット。
    
    Returns:
    dict: 選択されたグループ番号をキーとし、対応する配列IDリストを値とする辞書。
    """
    
    # 処理対象のグループ番号を読み込む
    selected_groups = set()
    with open(selected_groups_file, 'r') as file:
        for line in file:
            selected_groups.add(line.strip())

    # オルソロググループファイルからデータを読み込み、選択されたグループのみを辞書に登録
    groups = {}
    with open(orthologous_groups_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            group_number = parts[0]
            if group_number in selected_groups:  # 選択されたグループのみ処理
                # 菌種リストに含まれる配列IDのみをフィルタリングしてリストに格納
                ids = [id for id in parts[2:] if id[:5] in species_list]
                if ids:
                    groups[group_number] = ids  # 辞書にグループ番号とIDのリストを追加
    return groups


# 抽出した配列ペアをFASTAファイルに保存する関数
def extract_and_write_pairs(fasta_file, groups, output_dir):
    print("HERE")
    sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))  # FASTAファイルから配列を読み込み

    for group_number, ids in groups.items():
        print("HERE2")
        original_ids = ids.copy()  # オリジナルのIDリストをコピー
        i = 0
        while ids:
            if len(ids) == 1:  # IDが1つだけの場合、他のIDとペアを作る
                pair = [ids.pop()]
                other_id = random.choice([id for id in original_ids if id not in pair])
                pair.append(other_id)
            else:  # IDが2つ以上の場合、ランダムに2つ選択
                pair = random.sample(ids, 2)
                ids = [id for id in ids if id not in pair]

            species_names = '_'.join([seq_id[:5] for seq_id in pair])
            fasta_filename = f"{output_dir}/group{group_number}_{species_names}_{i+1}.fasta"
            
            # 新しいFASTAファイルに配列ペアを書き込み
            with open(fasta_filename, 'w') as fasta_file:
                for seq_id in pair:
                    if seq_id in sequences:
                        SeqIO.write(sequences[seq_id], fasta_file, "fasta")
            i += 1

# トレーニング用とテスト用のグループデータを処理する関数
def process_groups(train_groups_file, test_groups_file, oma_group_file, fasta_file, species_list, train_output_dir, test_output_dir):
    species_list = load_species_list(species_list_file)  # 菌種リストを読み込み
    print(species_list)
    train_groups = load_selected_groups(oma_group_file, train_groups_file, species_list)  # トレーニンググループを読み込み
    print(train_groups)
    test_groups = load_selected_groups(oma_group_file, test_groups_file, species_list)  # テストグループを読み込み
    
    # トレーニングデータのペアを抽出してFASTAファイルに保存
    extract_and_write_pairs(fasta_file, train_groups, train_output_dir)
    # テストデータのペアを抽出してFASTAファイルに保存
    extract_and_write_pairs(fasta_file, test_groups, test_output_dir)

# メイン処理
species_list_file = '/home/aca10223gf/workplace/data/OMA_database/prokaryotes_group.txt'  # 菌種リストファイルのパス
oma_group_file = '/home/aca10223gf/workplace/data/OMA_database/oma-groups.txt' # OMAgroupと配列IDのファイル
train_groups_file = '/home/aca10223gf/workplace/data/OMA_database/train_group.txt'  # トレーニンググループファイルのパス
test_groups_file = '/home/aca10223gf/workplace/data/OMA_database/test_group.txt'  # テストグループファイルのパス
fasta_file = '/home/aca10223gf/workplace/data/OMA_database/prokaryotes.cdna.fa'  # 配列が格納されたFASTAファイルのパス
train_output_dir = '/home/aca10223gf/workplace/data/OMA_database/full_data/train_fasta'  # トレーニングデータ出力ディレクトリ
test_output_dir = '/home/aca10223gf/workplace/data/OMA_database/full_data/test_fasta'  # テストデータ出力ディレクトリ

# トレーニング用とテスト用のグループデータを処理
process_groups(train_groups_file, test_groups_file, oma_group_file, fasta_file, species_list_file, train_output_dir, test_output_dir)
