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
def extract_and_write_pairs(fasta_file, groups, output_dir, mode):
    sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    for group_number, ids in groups.items():
        bacsu_ids = [id for id in ids if id[:5] == "BACSU"]
        idesa_ids = [id for id in ids if id[:5] == "IDESA"]
        other_ids = [id for id in ids if id[:5] not in ["BACSU", "IDESA"]]

        pairs = []
        if mode == "train":
            if bacsu_ids and idesa_ids:  # BACSUとIDESAの両方が含まれる場合
                pairs.append((bacsu_ids[0], idesa_ids[0]))  # 両方をペアとする
            elif bacsu_ids:  # BACSUのみ含まれる場合
                for bacsu_id in bacsu_ids:
                    if other_ids:
                        other_id = other_ids.pop(0)
                        pairs.append((bacsu_id, other_id))
            elif idesa_ids:  # IDESAのみ含まれる場合
                for idesa_id in idesa_ids:
                    if other_ids:
                        other_id = other_ids.pop(0)
                        pairs.append((other_id, idesa_id))
        elif mode == "test":
            if bacsu_ids and idesa_ids:  # BACSUとIDESAの両方が含まれる場合のみペアを作成
                pairs.append((bacsu_ids[0], idesa_ids[0]))

        # ペアをFASTAファイルに書き込む
        for i, (id1, id2) in enumerate(pairs):
            species_names = '_'.join([id1[:5], id2[:5]])
            fasta_filename = f"{output_dir}/group{group_number}_{species_names}_{i+1}.fasta"
            with open(fasta_filename, 'w') as fasta_file:
                if id1 in sequences:
                    SeqIO.write(sequences[id1], fasta_file, "fasta")
                if id2 in sequences:
                    SeqIO.write(sequences[id2], fasta_file, "fasta")

# トレーニング用とテスト用のグループデータを処理する関数
def process_groups(train_groups_file, test_groups_file, oma_group_file, fasta_file, species_list, train_output_dir, test_output_dir):
    species_list = load_species_list(species_list_file)  # 菌種リストを読み込み
    train_groups = load_selected_groups(oma_group_file, train_groups_file, species_list)  # トレーニンググループを読み込み
    test_groups = load_selected_groups(oma_group_file, test_groups_file, species_list)  # テストグループを読み込み
    
    # トレーニングデータのペアを抽出してFASTAファイルに保存
    extract_and_write_pairs(fasta_file, train_groups, train_output_dir, "train")
    # テストデータのペアを抽出してFASTAファイルに保存
    extract_and_write_pairs(fasta_file, test_groups, test_output_dir, "test")

# メイン処理
species_list_file = '/home/aca10223gf/workplace/data/OMA_database/prokaryotes_group.txt'  # 菌種リストファイルのパス
oma_group_file = '/home/aca10223gf/workplace/data/OMA_database/oma-groups.txt' # OMAgroupと配列IDのファイル
train_groups_file = '/home/aca10223gf/workplace/data/OMA_database/train_group.txt'  # トレーニンググループファイルのパス
test_groups_file = '/home/aca10223gf/workplace/data/OMA_database/test_group.txt'  # テストグループファイルのパス
fasta_file = '/home/aca10223gf/workplace/data/OMA_database/prokaryotes.cdna.fa'  # 配列が格納されたFASTAファイルのパス
train_output_dir = '/home/aca10223gf/workplace/data/OMA_database/BS_IS/train_fasta'  # トレーニングデータ出力ディレクトリ
test_output_dir = '/home/aca10223gf/workplace/data/OMA_database/BS_IS/test_fasta'  # テストデータ出力ディレクトリ

# トレーニング用とテスト用のグループデータを処理
process_groups(train_groups_file, test_groups_file, oma_group_file, fasta_file, species_list_file, train_output_dir, test_output_dir)
