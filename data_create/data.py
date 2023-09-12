# 必要なライブラリをインポート
import os
from Bio import Entrez, SeqIO

# RefSeq IDを使ってGenBankファイルをダウンロードする関数
def download_genbank_files(refseq_id):
    # Entrez.efetchを使って、指定したRefSeq IDのGenBankファイルをダウンロード
    with Entrez.efetch(db="nuccore", id=refseq_id, rettype="gbwithparts", retmode="text") as fetch_handle:
        # ダウンロードしたGenBankファイルをSeqRecordオブジェクトとして読み込む
        seq_record = SeqIO.read(fetch_handle, "gb")
    return seq_record

# CDS配列をFASTAファイルに保存する関数
def save_cds_to_fasta(strain_directory, seq_record):
    # SeqRecordオブジェクトのfeaturesプロパティから、CDSフィーチャーを1つずつ処理
    for feature in seq_record.features:
        # フィーチャーがCDSであり、かつprotein_idが含まれている場合
        if feature.type == "CDS" and "protein_id" in feature.qualifiers:
            # CDS配列をSeqRecordオブジェクトから抽出
            cds_sequence = feature.extract(seq_record.seq)
            # protein_idを取得
            cds_id = feature.qualifiers["protein_id"][0]
            # 保存するFASTAファイルの名前を作成
            filename = os.path.join(strain_directory, f"{cds_id}.fasta")
            # FASTAファイルを書き込む
            with open(filename, "w") as f:
                f.write(f">{cds_id}\n{cds_sequence}\n")

# 菌株名とRefSeqアクセッション番号のリスト
strain_names_and_refseq_ids = [
    ("Bacillus_subtilis_168", "NC_000964.3"),
    ("Bacillus_thuringiensis_ATCC_10792", "NZ_CM000753.1"),
    ("Bacillus_anthracis_Ames", "NC_003997.3"),
    ("Bacillus_cereus_ASM222028v1", "NC_004722.1"),
    ("Bacillus_clausii_assembly-7520-2", "NC_006582.1"),
    ("Bacillus_halodurans_C-125", "NC_002570.2"),
    ("Bacillus_coagulans_2-6", "NZ_CP009706.1"),
    ("Bacillus_licheniformis_DSM_13", "NC_006270.3"),
    ("Escherichia_coli_K-12_MG1655", "NC_000913.3"),
    ("Escherichia_coli_O157:H7_EDL933", "NC_002655.2"),
    ("Escherichia_coli_CFT073", "NC_004431.1"),
    ("Escherichia_fergusonii_ATCC_35469", "NC_011740.1"),
    ("Ideonella_sakaiensis_201-F6", "NZ_CP021194.1"),
    ("Ideonella_dechloratans", "NZ_LN831035.1"),
    ("Staphylococcus_aureus_NCTC_8325", "NC_007795.1"),
    ("Staphylococcus_epidermidis_RP62A", "NC_002976.3"),
    ("Streptococcus_pyogenes_MGAS5005", "NC_007297.1"),
    ("Streptococcus_pneumoniae_R6", "NC_003098.1"),
    ("Streptococcus_agalactiae_NEM316", "NC_004368.1"),
    ("Lactobacillus_plantarum_WCFS1", "NC_004567.2"),
    ("Lactobacillus_brevis_ATCC_367", "NC_008497.1"),
    ("Lactobacillus_rhamnosus_GG", "NC_013198.1"),
    ("Lactobacillus_casei_ATCC_334", "NC_008526.1"),
    ("Lactococcus_lactis_subsp_lactis_IL1403", "NC_002662.1"),
    ("Enterococcus_faecalis_V583", "NC_004668.1"),
    ("Enterococcus_faecium_DO", "NC_017960.1"),
    ("Pseudomonas_aeruginosa_PAO1", "NC_002516.2"),
    ("Pseudomonas_fluorescens_Pf-5", "NC_004129.6"),
]


# Entrezのメールアドレスを設定（重要）
Entrez.email = "your.email@example.com"

# ディレクトリの作成
base_directory = os.path.expanduser("~/data")
os.makedirs(base_directory, exist_ok=True)

# strain_names_and_refseq_idsリストから菌株名とRefSeq IDを1つずつ処理
for species_name, refseq_id in strain_names_and_refseq_ids:
    # 菌株名に基づいてディレクトリを作成
    strain_directory = os.path.join(base_directory, species_name.replace(" ", "_"))
    os.makedirs(strain_directory, exist_ok=True)

    # RefSeq IDを使ってGenBankファイルをダウンロード
    seq_record = download_genbank_files(refseq_id)
    # CDS配列をFASTAファイルに保存
    save_cds_to_fasta(strain_directory, seq_record)
