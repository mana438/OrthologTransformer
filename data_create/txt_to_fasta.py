from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# 入力ファイルと出力ファイルのパス
input_file = "/home/4/ux03574/workplace/job_results/20240809_215622/generate.fa"
output_file = "/home/4/ux03574/workplace/job_results/20240809_215622/generate_mod.fa"

# 入力ファイルを読み込み
with open(input_file, "r") as infile:
    dna_sequences = infile.read().strip().split("\n")

# SeqRecordリストを作成
seq_records = []
for i, sequence in enumerate(dna_sequences):
    record = SeqRecord(Seq(sequence), id=f"seq{i+1}", description="")
    seq_records.append(record)

# FASTA形式で書き出し
with open(output_file, "w") as outfile:
    SeqIO.write(seq_records, outfile, "fasta")

print(f"FASTA形式のファイルが以下に保存されました: {output_file}")
