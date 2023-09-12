# 必要なライブラリをインポート
import os
from Bio import Entrez, SeqIO

# RefSeq IDを使ってGenBankファイルをダウンロードする関数
def download_genbank_files(refseq_id):
    # Entrez.efetchを使って、指定したRefSeq IDのGenBankファイルをダウンロード
    try:
        fetch_handle = Entrez.efetch(db="nuccore", id=refseq_id, rettype="gbwithparts", retmode="text")
    except Exception as e:
        print(f"An error occurred while fetching data from Entrez: {e}")
        return None

    with fetch_handle as f:
        # ダウンロードしたGenBankファイルをSeqRecordオブジェクトとして読み込む
        seq_record = SeqIO.read(f, "gb")
        return seq_record


# CDS配列を菌株ごとのFASTAファイルに保存する関数
def save_cds_to_fasta(strain_directory_dna, strain_directory_aa, strain_name, seq_record):
    # 保存するFASTAファイルの名前を作成
    filename_dna = os.path.join(strain_directory_dna, f"{strain_name.replace(' ', '_')}_cds.fasta")
    filename_aa = os.path.join(strain_directory_aa, f"{strain_name.replace(' ', '_')}_cds.fasta")
    

    # FASTAファイルを書き込む
    with open(filename_dna, "w") as f_dna, open(filename_aa, "w") as f_aa:
        if seq_record is not None:
            # SeqRecordオブジェクトのfeaturesプロパティから、CDSフィーチャーを1つずつ処理
            for feature in seq_record.features:
                # フィーチャーがCDSであり、かつprotein_idが含まれている場合
                if feature.type == "CDS" and "protein_id" in feature.qualifiers:
                    # CDS配列をSeqRecordオブジェクトから抽出
                    
                    #DNA用
                    cds_sequence_dna = feature.extract(seq_record.seq)
                    #アミノ酸用
                    cds_sequence_aa = feature.qualifiers["translation"][0]

                    # protein_idを取得
                    cds_id = feature.qualifiers["protein_id"][0]
                    # FASTAファイルに書き込む
                    f_dna.write(f">{cds_id}\n{cds_sequence_dna}\n")
                    f_aa.write(f">{cds_id}\n{cds_sequence_aa}\n")

# 菌株名とRefSeqアクセッション番号のリスト
strain_names_and_refseq_ids = [
    ("Bacillus_subtilis_168", "NC_000964.3"),
    ("Bacillus_thuringiensis_ATCC_10792", "NZ_CM000753.1"),
    ("Bacillus_anthracis_AMES", "NC_003997.3"),
    ("Bacillus_cereus_ATCC_14579", "NC_004722.1"),
    ("Bacillus_licheniformis_ATCC_14580", "NC_006322.1"),
    ("Bacillus_megaterium_QM_B1551", "NC_014019.1"),
    ("Bacillus velezensis FZB42", "NC_009725.2"),
    ("Bacillus_halodurans_C-125", "NC_002570.2"),
    ('Bacillus_amyloliquefaciens', 'NC_014551.1'),
    ('Bacillus_atrophaeus', 'NC_014639.1'),
    ('Bacillus_coagulans', 'NC_016023.1'),
    ('Bacillus_licheniformis', 'NZ_CP014842.1'),

    # refseq idで持ってこれないので行わない
    # ("Ideonella_sakaiensis_201-F6", "GCA_001293525.1"),
    # ('Ideonella_dechloratans', 'GCF_021049305.1'),
    # ('Ideonella_paludis', 'GCF_018069865.1'),
    # ('Ideonella_benzenivorans', 'GCF_020387415.1'),
    # ('Ideonella_azotifigens', 'GCF_021043325.1'),
    # ('Ideonella_aquatica', 'GCF_018069755.1'),
    # ('Ideonella_livida', 'GCF_010499455.1'),
    # ('Ideonella_alba', 'GCF_018069875.1'),

    ("Cupriavidus_necator_H16", "NC_008313.1"),
    ("Pseudomonas_aeruginosa_PAO1", "NC_002516.2"),
    ("Pseudomonas_putida_KT2440", "NC_002947.4"),
    ("Pseudomonas_syringae_DC3000", "NC_004578.1"),
    ("Escherichia_coli_K-12_MG1655", "NC_000913.3"),
    ("Salmonella_enterica_serovar_Typhimurium_LT2", "NC_003197.2"),
    ("Lactobacillus_plantarum_WCFS1", "NC_004567.2"),
    ("Staphylococcus_aureus_NCTC_8325", "NC_007795.1"),
    ("Staphylococcus_epidermidis_ATCC_12228", "NC_004461.1"),
    ("Streptococcus_pneumoniae_TIGR4", "NC_003028.3"),
    ("Streptococcus_pyogenes_MGAS5005", "NC_007297.1"),
    ("Streptococcus_mutans_UA159", "NC_004350.2"),
    ("Streptococcus_thermophilus_LMG_18311", "NC_006448.1"),
    ('Exiguobacterium_sibiricum', 'NC_010556.1'),
    ('Geobacillus_thermodenitrificans', 'NC_009328.1'),
    ('Paenibacillus_polymyxa', 'NC_014483.1'),
    ('Nocardioides_sp_JS614', 'NC_008699.1'),
    ('Clostridium_perfringens', 'NC_008261.1'),
    ('Corynebacterium_kutscheri', 'NZ_LR134377.1'),
    ('Campylobacter_jejuni', 'NC_002163.1'),
    ('Klebsiella_pneumoniae', 'NC_009648.1'),
    ('Helicobacter_pylori_MT5135', 'NC_000915.1'),
    ('Listeria_monocytogenes', 'NC_003210.1'),
    ('Neisseria_meningitidis', 'NC_003112.2'),
    ('Streptococcus_agalactiae', 'NC_004116.1'),
    ('Treponema_pallidum', 'NC_000919.1'),
    ('Yersinia_pestis', 'NC_003143.1'),
    ('Salmonella_enterica', 'NC_003198.1'),
    ('Proteus_mirabilis', 'NC_010554.1'),
    ('Haemophilus_influenzae', 'NZ_CP007470.1'),
    ('Mycoplasma_pneumoniae', 'NC_000912.1'),
    ('Bordetella_pertussis', 'NC_002929.2'),
]

# Entrezのメールアドレスを設定（重要）
Entrez.email = "your.email@example.com"

# ディレクトリの作成
base_directory_dna = os.path.expanduser("/home/aca10223gf/workplace/data/CDS_dna")
base_directory_aa = os.path.expanduser("/home/aca10223gf/workplace/data/CDS_aa")

os.makedirs(base_directory_dna, exist_ok=True)
os.makedirs(base_directory_aa, exist_ok=True)

# strain_names_and_refseq_idsリストから菌株名とRefSeq IDを1つずつ処理
for species_name, refseq_id in strain_names_and_refseq_ids:
    # 菌株名に基づいてディレクトリを作成
    # strain_directory = os.path.join(base_directory, species_name.replace(" ", "_"))
    strain_directory_dna = base_directory_dna
    strain_directory_aa = base_directory_aa
    

    # RefSeq IDを使ってGenBankファイルをダウンロード
    print(species_name, refseq_id)
    seq_record = download_genbank_files(refseq_id)
    # CDS配列をFASTAファイルに保存
    save_cds_to_fasta(strain_directory_dna, strain_directory_aa, species_name, seq_record)
