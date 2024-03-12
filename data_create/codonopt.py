from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict

# ファスタファイルから配列を読み込む関数
def load_sequences_from_fasta(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# コドン使用頻度を計算する関数
def calculate_codon_usage(sequences):
    codon_usage = defaultdict(lambda: defaultdict(int))
    for sequence in sequences:
        for i in range(0, len(sequence), 3):
            codon = sequence[i:i+3]
            amino_acid = str(Seq(codon).translate())
            codon_usage[amino_acid][codon] += 1
    return codon_usage

# コドン最適化を行う関数
def optimize_codons(sequence, codon_usage):
    optimized_sequence = ""
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        amino_acid = str(Seq(codon).translate())
        # 最も頻繁に使用されるコドンを選択
        optimized_codon = max(codon_usage[amino_acid], key=codon_usage[amino_acid].get)
        optimized_sequence += optimized_codon
    return optimized_sequence

# アミノ酸配列をコドン配列に変換する関数
def translate_amino_acids_to_codons(amino_acid_sequence, codon_usage):
    codon_sequence = ""
    for amino_acid in amino_acid_sequence:
        if amino_acid in codon_usage:
            # 最も頻繁に使用されるコドンを選択
            codon = max(codon_usage[amino_acid], key=codon_usage[amino_acid].get)
            codon_sequence += codon
        else:
            raise ValueError(f"Unknown amino acid: {amino_acid}")
    return codon_sequence

# メイン処理
fasta_file = "/home/aca10223gf/workplace/data/CDS_dna/Bacillus_subtilis_168_cds.fasta"  # ファスタファイルのパス
sequences = load_sequences_from_fasta(fasta_file)
codon_usage = calculate_codon_usage(sequences)

# # 最適化したい配列
# target_sequence = "ATGAACTTTCCCCGCGCTTCCCGCCTGATGCAGGCCGCCGTTCTCGGCGGGCTGATGGCCGTGTCGGCCGCCGCCACCGCCCAGACCAACCCCTACGCCCGCGGCCCGAACCCGACAGCCGCCTCACTCGAAGCCAGCGCCGGCCCGTTCACCGTGCGCTCGTTCACCGTGAGCCGCCCGAGCGGCTACGGCGCCGGCACCGTGTACTACCCCACCAACGCCGGCGGCACCGTGGGCGCCATCGCCATCGTGCCGGGCTACACCGCGCGCCAGTCGAGCATCAAATGGTGGGGCCCGCGCCTGGCCTCGCACGGCTTCGTGGTCATCACCATCGACACCAACTCCACGCTCGACCAGCCGTCCAGCCGCTCGTCGCAGCAGATGGCCGCGCTGCGCCAGGTGGCCTCGCTCAACGGCACCAGCAGCAGCCCGATCTACGGCAAGGTCGACACCGCCCGCATGGGCGTGATGGGCTGGTCGATGGGCGGTGGCGGCTCGCTGATCTCGGCGGCCAACAACCCGTCGCTGAAAGCCGCGGCGCCGCAGGCCCCGTGGGACAGCTCGACCAACTTCTCGTCGGTCACCGTGCCCACGCTGATCTTCGCCTGCGAGAACGACAGCATCGCCCCGGTCAACTCGTCCGCCCTGCCGATCTACGACAGCATGTCGCGCAATGCGAAGCAGTTCCTCGAGATCAACGGTGGCTCGCACTCCTGCGCCAACAGCGGCAACAGCAACCAGGCGCTGATCGGCAAGAAGGGCGTGGCCTGGATGAAGCGCTTCATGGACAACGACACGCGCTACTCCACCTTCGCCTGCGAGAACCCGAACAGCACCCGCGTGTCGGACTTCCGCACCGCGAACTGCAGCTGA"
# optimized_sequence = optimize_codons(target_sequence, codon_usage)
# print("Optimized Sequence:", optimized_sequence)

# 最適化したいアミノ酸配列
amino_acid_sequence = "MSHILRAAVLAAMLLPLPSMAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPESRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWHSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSQNAKQFLEIKGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTAVSDFRTANCSLEHHHHHH"
codon_sequence = translate_amino_acids_to_codons(amino_acid_sequence, codon_usage)
print(codon_sequence)
optimized_codon_sequence = optimize_codons(codon_sequence, codon_usage)

print("Optimized Codon Sequence:", optimized_codon_sequence)
