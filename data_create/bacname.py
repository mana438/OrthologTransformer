from Bio import Entrez, SeqIO
import os
import subprocess
import zipfile
import shutil
import requests
import datetime
import re
from Bio.SeqRecord import SeqRecord
import random


def is_before_2023_12_31(submission_date_str):
    from datetime import datetime
    # 提出日をdatetimeオブジェクトに変換
    submission_date = datetime.strptime(submission_date_str, "%Y/%m/%d %H:%M")

    # 比較対象の日付（2023年12月31日）を定義
    comparison_date = datetime(2023, 12, 31, 23, 59, 59)

    # 提出日が2023年12月31日より前であるかどうかを判断
    return submission_date <= comparison_date


def extract_first_two_words(name):
    """菌株名から最初の2単語を抽出する関数"""
    return ' '.join(name.split()[:2])

def download_and_process_genome(genbank_id, dna_dir, aa_dir, formatted_species_name):
    get_bact = False

    # 菌種名の処理
    formatted_species_name = formatted_species_name.replace(" ", "_")

    # curlコマンドを構築
    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{genbank_id}/download?include_annotation_type=CDS_FASTA"
    zip_file = "genome_data.zip"

    # curlコマンドを実行してデータをダウンロード
    subprocess.run(["curl", url, "--output", zip_file], check=True)

    # zipファイルを解凍
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("temp_genome")

    # fastaファイルを探し出す
    for root, dirs, files in os.walk("temp_genome"):
        for file in files:
            if file.endswith(".fna"):
                fasta_file = os.path.join(root, file)

                # ファイルを読み込み、条件をチェック
                dna_sequences = []
                aa_sequences = []
                for record in SeqIO.parse(fasta_file, "fasta"):
                    # '_cds_' 以降から空白文字までを抜き出す
                    match = re.search(r'_cds_([^ ]+)', record.description)
                    if match:
                        new_id = match.group(1)
                        record.id = new_id
                        record.description = new_id
                        dna_sequences.append(record)

                        # アミノ酸配列の翻訳
                        aa_seq = record.seq.translate()
                        # 新しいSeqRecordオブジェクトを作成
                        aa_record = SeqRecord(aa_seq, id=new_id, description=new_id)
                        aa_sequences.append(aa_record)
                        
                    else:
                        print("Warning: No match found in record description.")
                        print(record.description)
                    

                 # 配列数が1500以上あるか確認
                if len(dna_sequences) >= 1500:
                    get_bact = True
                    # DNA配列を保存
                    dna_path = os.path.join(dna_dir, f"{formatted_species_name}.fa")
                    SeqIO.write(dna_sequences, dna_path, "fasta")

                    # アミノ酸配列を保存
                    aa_path = os.path.join(aa_dir, f"{formatted_species_name}.fa")
                    SeqIO.write(aa_sequences, aa_path, "fasta")
                break

    # 一時ディレクトリとzipファイルを削除
    shutil.rmtree("temp_genome")
    os.remove(zip_file)

    return get_bact


def search_strains(query, dna_dir, aa_dir, max_results=1000000):
    Entrez.email = "your.email@example.com"  # 自分のメールアドレスを設定
    handle = Entrez.esearch(db="assembly", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    # esearchで得られたIDを使用してesummaryを実行
    summary_handle = Entrez.esummary(db="assembly", id=",".join(ids))
    summary_record = Entrez.read(summary_handle)

    # 閾値の設定
    contig_n50_threshold = 50000  # Contig N50の閾値
    coverage_threshold = 50.0    # Coverageの閾値

    best_records = {}

    # 条件を満たす最良のレコードを集める
    for docsum in summary_record['DocumentSummarySet']['DocumentSummary']:
        organism_name = docsum['Organism']
        shortened_organism_name = extract_first_two_words(organism_name)
        contig_n50 = int(docsum.get('ContigN50', 0))
        coverage_str = docsum.get('Coverage', '0')
        submission_date = docsum.get('SubmissionDate', 'N/A')

        try:
            coverage = float(coverage_str) if coverage_str else 0.0
        except ValueError:
            coverage = 0.0

        # 閾値を満たすかどうかを確認
        if contig_n50 >= contig_n50_threshold and coverage >= coverage_threshold and is_before_2023_12_31(submission_date):
            if shortened_organism_name not in best_records or (contig_n50 > best_records[shortened_organism_name][1] and coverage > best_records[shortened_organism_name][2]):
                best_records[shortened_organism_name] = (docsum, contig_n50, coverage)

    # ランダムに20個選ぶ（レコード数が10個未満の場合は全て選ぶ）
    selected_records = random.sample(list(best_records.values()), min(100, len(best_records)))

    unique_gca_ids = {}

    # 選んだレコードを処理する
    for docsum, contig_n50, coverage in selected_records:
        if 'AssemblyAccession' in docsum and 'Organism' in docsum:
            organism_name = docsum['Organism']
            shortened_organism_name = extract_first_two_words(organism_name)
            gca_id = docsum['AssemblyAccession']
            submission_date = docsum.get('SubmissionDate', 'N/A')
            last_update = docsum.get('LastUpdateDate', 'N/A')

            if shortened_organism_name not in unique_gca_ids or (contig_n50 >= unique_gca_ids[shortened_organism_name][1] and coverage >= unique_gca_ids[shortened_organism_name][2]):
                try:
                    get_bact = download_and_process_genome(gca_id, dna_dir, aa_dir, shortened_organism_name)
                    if get_bact:
                        unique_gca_ids[shortened_organism_name] = (gca_id, contig_n50, coverage, organism_name, submission_date, last_update)
                except Exception as e:
                    print(f"Error with GCA ID {gca_id}, {organism_name}, {submission_date}: {e}")    

   # ファイルに書き込み
    with open(file_name, 'a') as file:
        for shortened_name, (gca_id, contig_n50, coverage, organism_name, submission_date, last_update) in unique_gca_ids.items():
            output_line = f"Organism: {organism_name}, GCA ID: {gca_id}, Contig N50: {contig_n50}, Coverage: {coverage}, Submission Date: {submission_date}, Last Update: {last_update}"
            print(output_line)
            file.write(output_line + "\n")

    return unique_gca_ids

# 現在の日時を取得し、ファイル名に使用する形式に変換
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ディレクトリの作成
base_dir = f"/home/aca10223gf/workplace/data/{current_time}"
dna_dir = os.path.join(base_dir, "CDS_dna")
aa_dir = os.path.join(base_dir, "CDS_aa")

os.makedirs(dna_dir, exist_ok=True)
os.makedirs(aa_dir, exist_ok=True)

file_name = f"/home/aca10223gf/workplace/data/{current_time}/info.txt"

# 菌種ベース
# querys = [
#     "Ideonella[Organism]", "Escherichia[Organism]", "Staphylococcus[Organism]", 
#     "Streptococcus[Organism]", "Bacillus[Organism]", "Pseudomonas[Organism]", 
#     "Clostridium[Organism]", "Mycobacterium[Organism]", "Lactobacillus[Organism]", 
#     "Salmonella[Organism]", "Helicobacter[Organism]", "Listeria[Organism]", 
#     "Vibrio[Organism]", "Campylobacter[Organism]", "Bordetella[Organism]", 
#     "Corynebacterium[Organism]", "Enterococcus[Organism]", "Neisseria[Organism]", 
#     "Haemophilus[Organism]", "Legionella[Organism]", "Bacteroides[Organism]", 
#     "Klebsiella[Organism]", "Proteus[Organism]", "Yersinia[Organism]", 
#     "Shigella[Organism]", "Acinetobacter[Organism]", "Burkholderia[Organism]", 
#     "Francisella[Organism]", "Brucella[Organism]", "Rickettsia[Organism]", 
#     "Chlamydia[Organism]", "Synechococcus[Organism]", "Rhodobacter[Organism]", 
#     "Serratia[Organism]", "Enterobacter[Organism]", "Erwinia[Organism]", 
#     "Nocardia[Organism]", "Actinomyces[Organism]", "Micrococcus[Organism]", 
#     "Anabaena[Organism]", "Nitrosomonas[Organism]", "Nitrobacter[Organism]", 
#     "Sulfolobus[Organism]", "Thermus[Organism]", "Deinococcus[Organism]", 
#     "Arthrobacter[Organism]", "Propionibacterium[Organism]", "Xanthomonas[Organism]", 
#     "Flavobacterium[Organism]", "Rhizobium[Organism]", "Agrobacterium[Organism]", 
#     "Nitrosospira[Organism]", "Nitrospira[Organism]", "Chloroflexus[Organism]", 
#     "Chlorobium[Organism]", "Streptomyces[Organism]", "Thermoactinomyces[Organism]", 
#     "Methanobacterium[Organism]", "Methanosarcina[Organism]", "Halobacterium[Organism]", 
#     "Halococcus[Organism]", "Lactococcus[Organism]", "Leuconostoc[Organism]", 
#     "Weissella[Organism]", "Pediococcus[Organism]", "Aeromonas[Organism]", 
#     "Arcobacter[Organism]", "Wolbachia[Organism]", "Bifidobacterium[Organism]", 
#     "Gardnerella[Organism]", "Treponema[Organism]", "Borrelia[Organism]", 
#     "Leptospira[Organism]", "Mycoplasma[Organism]", "Ureaplasma[Organism]", 
#     "Spiroplasma[Organism]", "Veillonella[Organism]", "Fusobacterium[Organism]", 
#     "Prevotella[Organism]", "Porphyromonas[Organism]", "Tannerella[Organism]", 
#     "Capnocytophaga[Organism]", "Gemella[Organism]", "Rothia[Organism]", 
#     "Moraxella[Organism]", "Acetobacter[Organism]", "Gluconobacter[Organism]", 
#     "Komagataeibacter[Organism]", "Azotobacter[Organism]", "Beijerinckia[Organism]", 
#     "Derxia[Organism]", "Caulobacter[Organism]", "Hyphomicrobium[Organism]", 
#     "Sphingomonas[Organism]", "Zymomonas[Organism]", "Rhodospirillum[Organism]", 
#     "Rhodopseudomonas[Organism]", "Rhodomicrobium[Organism]", "Rhodoplanes[Organism]", 
#     "Pirellula[Organism]", "Planctomyces[Organism]"
# ]

# querys = ["Bacillus subtilis[Organism]", "Ideonella [Organism]"]

# 菌株ベース
querys = [
    "Acidovorax delafieldii[Organism]", "Albitalea terrae[Organism]",
    "Alphaproteobacteria bacterium[Organism]", "Aquabacterium fontiphilum[Organism]",
    "Aquabacterium lacunae[Organism]", "Aquabacterium terrae[Organism]",
    "Bacillus cereus[Organism]", "Bacillus halotolerans[Organism]",
    "Bacillus rugosus[Organism]", "Bacillus siamensis[Organism]",
    "Bacillus spizizenii[Organism]", "Bacillus stercoris[Organism]",
    "Bacillus subtilis[Organism]", "Bacillus tequilensis[Organism]",
    "Bacillus thuringiensis[Organism]", "Bacillus velezensis[Organism]",
    "Burkholderiales bacterium[Organism]", "Caldimonas brevitalea[Organism]",
    "Caldimonas caldifontis[Organism]", "Clostridium beijerinckii[Organism]",
    "Clostridium estertheticum[Organism]", "Ideonella oryzae[Organism]",
    "Ideonella paludis[Organism]", "Ideonella sakaiensis[Organism]",
    "Iodobacter violacea[Organism]", "Oleispira antarctica[Organism]",
    "Oxalobacteraceae bacterium[Organism]", "Paenacidovorax monticola[Organism]",
    "Paraburkholderia aspalathi[Organism]", "Paucibacter toxinivorans[Organism]",
    "Piscinibacter defluvii[Organism]", "Piscinibacter lacus[Organism]",
    "Pseudacidovorax intermedius[Organism]", "Pseudorhodoferax aquiterrae[Organism]",
    "Pseudorhodoferax soli[Organism]", "Rhizobacter gummiphilus[Organism]",
    "Rhodocyclaceae bacterium[Organism]", "Schlegelella aquatica[Organism]",
    "Schlegelella koreensis[Organism]", "Scleromatobacter humisilvae[Organism]",
    "Sphaerotilus hippei[Organism]", "Streptococcus dysgalactiae[Organism]",
    "Streptococcus infantarius[Organism]", "Streptococcus ruminantium[Organism]",
    "Sulfuriferula multivorans[Organism]", "Thermobifida alba[Organism]"
]



for query in querys:
    unique_gca_ids = search_strains(query, dna_dir, aa_dir)