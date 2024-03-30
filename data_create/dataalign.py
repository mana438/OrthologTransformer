import os
import re
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio import SeqIO

def align_sequences(file_list, output_dir, use_gap=True):
    # BLOSUM62の置換行列をロードします
    blosum62 = substitution_matrices.load("BLOSUM62")

    # PairwiseAlignerオブジェクトを作成します
    aligner = Align.PairwiseAligner()

    # BLOSUM62をスコア行列として設定します
    aligner.substitution_matrix = blosum62

    # ギャップの開始（作成）と延長に対するペナルティを設定します
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    aligner.mode = 'global'

    for fasta_file in file_list:
        # FASTAファイルから配列を読み込みます
        seq_records = list(SeqIO.parse(fasta_file, "fasta"))
        seq1 = seq_records[0].seq
        seq2 = seq_records[1].seq
        seq1_id = seq_records[0].id
        seq2_id = seq_records[1].id

        # DNA配列をアミノ酸配列に変換します
        protein_seq1 = seq1.translate()
        protein_seq2 = seq2.translate()

        # アミノ酸配列をアラインメントします
        alignments = aligner.align(protein_seq1, protein_seq2)

        # 最良のアラインメントを取得します
        best_alignment = alignments[0]
        aligned_seq1 = str(best_alignment[0])
        aligned_seq2 = str(best_alignment[1])

        # DNA配列の変更を計算
        modified_dna_seq1 = ""
        modified_dna_seq2 = ""
        i, j = 0, 0

        for aa1, aa2 in zip(best_alignment[0], best_alignment[1]):
            if use_gap:
                if aa1 == '-':
                    modified_dna_seq1 += "---"
                else:
                    modified_dna_seq1 += str(seq1[i:i+3])
                    i += 3
                if aa2 == '-':
                    modified_dna_seq2 += "---"
                else:
                    modified_dna_seq2 += str(seq2[j:j+3])
                    j += 3
            else:
                if aa1 == '-' and aa2 != '-':
                    j += 3
                elif aa1 != '-' and aa2 == '-':
                    i += 3
                else:
                    modified_dna_seq1 += str(seq1[i:i+3])
                    modified_dna_seq2 += str(seq2[j:j+3])
                    i += 3
                    j += 3

        # 出力ファイル名を生成
        output_file = os.path.join(output_dir, os.path.basename(fasta_file))

        # 結果をファイルに書き込みます
        with open(output_file, "w") as f:
            f.write(f">{seq1_id}\n{modified_dna_seq1}\n")
            f.write(f">{seq2_id}\n{modified_dna_seq2}\n")

# 使用例
input_dir = "/home/aca10223gf/workplace/data/OMA_database/full_data/train_fasta/"
output_dir = "/home/aca10223gf/workplace/data/OMA_database/full_data_align/train_fasta"

# 入力ディレクトリ内のFASTAファイルを取得
fasta_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".fasta")]

align_sequences(fasta_files, output_dir, False)