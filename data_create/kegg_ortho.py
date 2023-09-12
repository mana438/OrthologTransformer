import os
from Bio.KEGG import REST
import time

# 代表的な菌株のリスト
representative_strains = [
    "Bacillus+subtilis",
    "Bacillus+thuringiensis",
    "Ideonella+sakaiensis",
    "Bacillus+anthracis",
    "Bacillus+cereus",
    "Bacillus+licheniformis",
    "Bacillus+megaterium",
    "Bacillus+velezensis",
    "Bacillus+halodurans",
    "Cupriavidus+necator",
    "Pseudomonas+aeruginosa",
    "Pseudomonas+putida",
    "Pseudomonas+syringae",
    "Escherichia+coli",
    "Salmonella+enterica",
    "Lactobacillus+plantarum",
    "Staphylococcus+aureus",
    "Staphylococcus+epidermidis",
    "Streptococcus+pneumoniae",
    "Streptococcus+pyogenes",
    "Streptococcus+mutans",
    "Streptococcus+thermophilus",
]

# 出力ディレクトリを作成
output_dir = "/home/aca10223gf/workplace/data/CDS_ortho/"
os.makedirs(output_dir, exist_ok=True)

# 各菌種の代表菌株を選択し、そのCDS配列を抽出
for strain_name in representative_strains:
    print(f"Processing {strain_name}")

    # 菌種名を指定してKEGG Organismデータベースから菌株を取得
    strain_list = REST.kegg_find("genome", strain_name).read()
    strain_ids = [line.split("\t")[0].split(":")[1] for line in strain_list.split("\n") if line]

    # この例では、最初に見つかった菌株を代表菌株として選択します
    representative_strain_id = strain_ids[0]

    # 代表菌株の遺伝子IDを取得
    genes = REST.kegg_list(representative_strain_id).read()
    gene_ids = [line.split('\t')[0] for line in genes.split('\n') if line]

    # 各遺伝子IDに対して、オルソロググループとCDS配列を取得し、ファイルに出力
    for gene_id in gene_ids:
        time.sleep(0.1)  # ウェイトを追加
        ortholog_group_response = REST.kegg_link("ko", gene_id).read()
        ortholog_groups = [line.split("\t")[1] for line in ortholog_group_response.split("\n") if line]

        # オルソロググループが見つかった場合
        if ortholog_groups:
            ortholog_group = ortholog_groups[0]

            try:
                cds_sequence = REST.kegg_get(gene_id, "ntseq").read()
            except Exception as e:
                print(f"Error fetching CDS sequence for {gene_id}: {e}")
                continue

            # オルソロググループごとのファイルにCDS配列を追加
            output_folder = os.path.join(output_dir, ortholog_group)
            os.makedirs(output_folder, exist_ok=True)
            output_filename = os.path.join(output_folder, f"{representative_strain_id}_{ortholog_group}.fasta")

            with open(output_filename, "a") as output_file:
                output_file.write(f">{gene_id}\n{cds_sequence}\n")