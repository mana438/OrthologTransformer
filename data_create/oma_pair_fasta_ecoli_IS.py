#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDESA → 多数の E. coli 亜種 (40 種以上) について
train/test 用 FASTA ファイルを自動生成するスクリプト

2025-05-15  Akiyama Lab
"""

from Bio import SeqIO   # Biopython の FASTA 解析モジュール
import os               # ファイル／ディレクトリ操作用標準ライブラリ

# =========================================================
# 1. ユーティリティ関数
# =========================================================

def load_species_list(species_list_file):
    """
    指定したファイルから菌種 (5 文字の OMA コード) を読み込み
    セット型で返す簡易ヘルパー

    Parameters
    ----------
    species_list_file : str
        1 行 1 コードで書かれたファイルへのパス

    Returns
    -------
    set[str]
        読み込んだ菌種コードの集合
    """
    with open(species_list_file) as f:
        return {line.strip() for line in f}


def load_selected_groups(oma_groups_file, selected_groups_file, species_set):
    """
    ・処理対象のグループ番号 (train / test) を読み込み  
    ・OMA のオルソロググループファイルを走査し  
      └ グループ番号が対象一覧に含まれ  
      └ かつ 5 文字菌種コードが species_set 内にある配列だけを抽出  
    して {group_no: [seq_id, ...]} の辞書を返す

    Notes
    -----
    OMA ファイルは TSV で  
        col0 : group_no  
        col1 : （空カラムなど）  
        col2～ : seq_id  
    の形式を想定。
    """
    # --- 対象グループ番号を読み込む ---
    with open(selected_groups_file) as f:
        target_groups = {line.strip() for line in f}

    groups = {}
    # --- OMA groups ファイルを 1 行ずつ処理 ---
    with open(oma_groups_file) as f:
        for line in f:
            cols = line.rstrip().split('\t')
            gno  = cols[0]
            if gno not in target_groups:          # 対象外のグループはスキップ
                continue

            # cols[2:] には「配列 ID (例: IDESA1234)」が並ぶ  
            # 先頭 5 文字 = 菌種コード でフィルタリング
            ids = [sid for sid in cols[2:] if sid[:5] in species_set]

            # 少なくとも 1 本は残っている場合のみ登録
            if ids:
                groups[gno] = ids
    return groups


def extract_and_write_pairs(fasta_file, groups, output_dir,
                            mode, input_species, output_species):
    """
    1 グループ → 0 以上のペア → 1 FASTA ファイル
    を生成するメイン処理

    Parameters
    ----------
    fasta_file : str
        全配列収録 FASTA（cdna.fa など）
    groups : dict[str, list[str]]
        {group_no: [seq_id, …]} 形式の辞書
    output_dir : str
        train_fasta / test_fasta ディレクトリ
    mode : {"train", "test"}
        トレーニング用 or テスト用でペア生成ルールが異なる
    input_species : str
        例: "IDESA"
    output_species : str
        例: "ECO1A"
    """
    # --- FASTA から全配列を辞書に読み込み（高速アクセス用） ---
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    # 出力ディレクトリを作成（既に存在していてもエラーにならない）
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 各グループを処理 ----------
    for gno, ids in groups.items():

        # 対象菌種ごとに ID を分類
        in_ids   = [sid for sid in ids if sid[:5] == input_species]   # IDESA
        out_ids  = [sid for sid in ids if sid[:5] == output_species]  # ECO***
        others   = [sid for sid in ids if sid[:5] not in {input_species, output_species}]

        pairs = []  # [(seq_id1, seq_id2), …] を格納するリスト

        # === ペア生成ルール ===
        if mode == "train":
            # --- 最優先: 両方そろっていれば 1 ペアだけ ---
            if in_ids and out_ids:
                pairs.append((out_ids[0], in_ids[0]))
            # --- 片方のみの場合は残りを other から引き当て ---
            elif out_ids:
                for oid in out_ids:
                    if others:
                        pairs.append((oid, others.pop(0)))
            elif in_ids:
                for iid in in_ids:
                    if others:
                        pairs.append((others.pop(0), iid))

        elif mode == "test":
            # テストは「両種そろう場合のみ 1 ペア」
            if in_ids and out_ids:
                pairs.append((out_ids[0], in_ids[0]))

        # === FASTA 書き出し ===
        for i, (sid1, sid2) in enumerate(pairs, 1):
            # 例: group00123_ECO1A_IDESA_1.fasta
            fname = f"group{gno}_{sid1[:5]}_{sid2[:5]}_{i}.fasta"
            fpath = os.path.join(output_dir, fname)

            with open(fpath, "w") as fh:
                # 双方とも辞書にあるはずだが念のため存在チェック
                if sid1 in seq_dict:
                    SeqIO.write(seq_dict[sid1], fh, "fasta")
                if sid2 in seq_dict:
                    SeqIO.write(seq_dict[sid2], fh, "fasta")


def process_groups(train_groups_file, test_groups_file, oma_groups_file,
                   fasta_file, species_list_file,
                   train_outdir, test_outdir,
                   input_species, output_species):
    """
    1 つの <<IDESA, ECO***>> 組み合わせについて
    ・train グループ → train_fasta  
    ・test  グループ → test_fasta  
    を生成するラッパ関数
    """
    sp_set = load_species_list(species_list_file)

    train_g = load_selected_groups(oma_groups_file, train_groups_file, sp_set)
    test_g  = load_selected_groups(oma_groups_file, test_groups_file,  sp_set)

    extract_and_write_pairs(fasta_file, train_g, train_outdir,
                            "train", input_species, output_species)
    extract_and_write_pairs(fasta_file, test_g,  test_outdir,
                            "test",  input_species, output_species)


# =========================================================
# 2. パス設定（自分の環境に合わせて適宜変更）
# =========================================================
ROOT = '/gs/bs/tgh-25IAF/akiyama/data/OMA_database'

species_list_file = f'{ROOT}/prokaryotes_group.txt'
oma_groups_file   = f'{ROOT}/oma-groups.txt'
train_groups_file = f'{ROOT}/train_group.txt'
test_groups_file  = f'{ROOT}/test_group.txt'
fasta_file        = f'{ROOT}/prokaryotes.cdna.fa'

# "BS_IS" という文字列は置換用のダミー
base_root = f'{ROOT}/BS_IS'

# =========================================================
# 3. 生成対象の菌種コード一覧
# =========================================================
input_species = "IDESA"  # 固定
output_species_list = [
    "ECO10",
    "ECO1A","ECO1E","ECO24","ECO26","ECO27","ECO44","ECO45","ECO55","ECO57",
    "ECO5E","ECO5T","ECO7I","ECO81","ECO8A","ECO8N","ECOAB","ECOBB","ECOBD",
    "ECOBR","ECOBW","ECOC1","ECOC2","ECOCB","ECOD1","ECODH","ECOH1","ECOHS",
    "ECOK1","ECOKI","ECOKO","ECOL5","ECOL6","ECOLC","ECOLI","ECOLU","ECOLW",
    "ECOLX","ECOS5","ECOSE","ECOSM","ECOUM","ECOUT"
]

# =========================================================
# 4. メインループ
# =========================================================
for out_sp in output_species_list:
    # 例: BS_IS → IDESA_ECO1A に置換
    pair_root = base_root.replace("BS_IS", f"{out_sp}_{input_species}")
    train_out = os.path.join(pair_root, "train_fasta")
    test_out  = os.path.join(pair_root, "test_fasta")

    # 1 種類ずつ train / test FASTA を生成
    process_groups(train_groups_file, test_groups_file, oma_groups_file,
                   fasta_file, species_list_file,
                   train_out, test_out,
                   input_species, out_sp)

print("✅ すべての IDESA→ECO*** データセットの生成が完了しました。")
