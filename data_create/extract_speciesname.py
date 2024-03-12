def extract_first_five_unique_from_fasta(input_file, output_file):
    unique_ids = set()  # 重複を避けるためにセットを使用
    with open(input_file, 'r') as fasta, open(output_file, 'w') as output:
        for line in fasta:
            if line.startswith('>'):  # ID行を識別
                first_five = line[1:6]
                if first_five not in unique_ids:  # まだセットに含まれていない場合
                    unique_ids.add(first_five)  # セットに追加
                    output.write(first_five + '\n')  # ファイルに書き込み

# 実行例
# 'input.fasta'は読み込むFASTAファイルの名前
# 'output.txt'は出力ファイルの名前
extract_first_five_unique_from_fasta('/home/aca10223gf/workplace/data/OMA_database/prokaryotes.cdna.fa', '/home/aca10223gf/workplace/data/OMA_database/prokaryotes_group.txt')
