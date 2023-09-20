import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
import random
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
import time

class Vocab:
    def __init__(self, tokens):
        self.token_to_index = {}
        self.index_to_token = {}
        self.build_vocab(tokens)

    def build_vocab(self, tokens):
        for token in tokens:
            if token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token

    def __len__(self):
        return len(self.token_to_index)
    
    def __getitem__(self, token):
        return self.token_to_index[token]

class OrthologDataset(Dataset):
    def __init__(self, fasta_dir):
        self.data = []
        self.ortholog_groups = set()
        self.fasta_dir = fasta_dir
        # すべての菌種名のリストを取得
        species_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(self.fasta_dir) if file_name.endswith(".fasta")]
        # すべてのコドンのリスト
        codons = [f"{x}{y}{z}" for x in "ACGT" for y in "ACGT" for z in "ACGT"]
        gap = ["---"]

        # すべてのアミノ酸のリスト
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
        #　すべての塩基リスト
        dna = ["A", "C", "G", "T"]
        # 特殊文字
        spc = ['<pad>', '<bos>', '<eos>']

        # コドンと菌種名の両方を含むトークンのリストを作成
        tokens =  spc + codons + species_names
        self.vocab = Vocab(tokens)
        # アミノ酸も追加8/14
        tokens_amino =  spc + amino_acids + species_names        
        self.vocab_amino = Vocab(tokens_amino)
        # DNAも追加8/31
        tokens_dna =  spc + dna + species_names        
        self.vocab_dna = Vocab(tokens_dna)

        self.vocab_target = spc + codons
        self.vocab_target_amino = spc + amino_acids
        self.vocab_target_dna = spc + dna

        # 64次元の全てが0であるリスト
        zero_list = [0] * 64
        self.codon_dict = {}
        # 各キーに対して64次元の全てが0であるリストを値として設定
        for key in codons:
            self.codon_dict[key] = zero_list.copy()
        self.num_dict = {}
        for i, key in enumerate(codons):
            self.num_dict[key] = i



    def load_data(self, ortholog_files, exclude_test_group=False):
        # オルソログ関係ファイルを1つずつ読み込む
        for ortholog_file in ortholog_files:
            with open(ortholog_file, "r") as f:
                ortholog_data = f.readlines()
            seq1_species = ortholog_data[0].split("\t")[1].strip()
            seq2_species = ortholog_data[0].split("\t")[2].strip()

            # 各オルソログ関係を解析し、ペアを作る
            for line in ortholog_data[1:]:
                ortholog_group, seq1, seq2 = line.strip().split("\t")
                if exclude_test_group and ortholog_group in self.test_groups:
                    continue

                self.ortholog_groups.add(ortholog_group)
                seq1_ids = seq1.split(", ")
                seq2_ids = seq2.split(", ")
                # 出力配列
                for seq1_id in seq1_ids:
                    seq1_fasta_file = os.path.join(self.fasta_dir, f"{seq1_species}.fasta")
                    seq1_seq = self.read_fasta(seq1_fasta_file, seq1_id)
                    if seq1_seq == None:
                        continue              

                    scores = []
                    seq2_seqs = []
                    seq2_seqs_mod = []
                    seq1_seqs_mod = []
                    gap_rates = []
                    
                    # 入力配列
                    for seq2_id in seq2_ids:
                        seq2_fasta_file = os.path.join(self.fasta_dir, f"{seq2_species}.fasta")
                        seq2_seq = self.read_fasta(seq2_fasta_file, seq2_id)
                        if not seq2_seq:
                            continue
                        try:
                            gap_rate, alignment_score, modified_dna_seq1, modified_dna_seq2 = self.align(seq1_seq, seq2_seq)
                            seq2_seqs_mod.append(modified_dna_seq2)
                            seq1_seqs_mod.append(modified_dna_seq1)
                            scores.append(alignment_score)
                            gap_rates.append(gap_rate)
                        except ValueError as e:
                            print("value error")
                            # scores.append(0)
                            # gap_rate = 0

                    if seq1_seq == None or not bool(seq2_seqs_mod):
                        continue

                    # 入力配列の中で最も出力配列と類似度を高いものを選定    
                    seq2_seq_mod = seq2_seqs_mod[scores.index(max(scores))]
                    seq1_seq_mod = seq1_seqs_mod[scores.index(max(scores))]
                    gap_rate = gap_rates[scores.index(max(scores))]

                    seq2seqs = []   
                    seq1seqs = []

                    # length = 102
                    # if len(seq2_seq_mod) > length:
                    #     for _ in range(30):
                    #         start_index = random.randint(0, len(seq2_seq_mod) - length)
                    #         substring2 = seq2_seq_mod[start_index:start_index + length]
                    #         seq2seqs.append(substring2) 
                    #         substring1 = seq1_seq_mod[start_index:start_index + length]
                    #         seq1seqs.append(substring1)
                    # else:
                    #     seq2seqs = [seq2_seq_mod]
                    #     seq1seqs = [seq1_seq_mod]

                    seq2seqs = [seq2_seq_mod]
                    seq1seqs = [seq1_seq_mod]
                    # pro
                    for seq1_seq, seq2_seq in zip(seq1seqs, seq2seqs):
                        # ペアが有効である場合、データに追加する
                        if self.is_valid_pair(seq1_seq, seq2_seq, gap_rate) or ortholog_group == "OG999999":
                            seq1_codons = self.convert_to_codons(seq1_seq, add_species=False)
                            seq1_proteins = self.convert_to_amino(Seq(seq1_seq).translate(), add_species=False)
                            seq1_dna = self.convert_to_dna(seq1_seq, add_species=False)
                            
                            seq2_codons = self.convert_to_codons(seq2_seq, seq1_species, seq2_species, add_species=True)
                            seq2_proteins = self.convert_to_amino(Seq(seq2_seq).translate(), seq1_species, seq2_species, add_species=True)
                            seq2_dna = self.convert_to_dna(seq2_seq, seq1_species, seq2_species, add_species=True)
                            
                            self.data.append((ortholog_group, seq1_codons, seq1_proteins, seq1_dna, seq2_codons, seq2_proteins, seq2_dna))

        # print(self.codon_dict)



    def convert_to_codons(self, seq, seq1_species=None, seq2_species=None, add_species=True):
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]

        if add_species:
            seq1_species_index = self.vocab[seq1_species]
            seq2_species_index = self.vocab[seq2_species]
            codon_seq = [self.vocab['<bos>']] + [seq1_species_index, seq2_species_index] + [self.vocab[codon] for codon in codons] + [self.vocab['<eos>']]
        else:
            codon_seq = [self.vocab['<bos>']] + [self.vocab[codon] for codon in codons] + [self.vocab['<eos>']]

        return codon_seq

    def convert_to_amino(self, seq, seq1_species=None, seq2_species=None, add_species=True):
        if add_species:
            seq1_species_index = self.vocab_amino[seq1_species]
            seq2_species_index = self.vocab_amino[seq2_species]
            amino_seq = [self.vocab_amino['<bos>']] + [seq1_species_index, seq2_species_index] + [self.vocab_amino[amino] for amino in list(seq)] + [self.vocab_amino['<eos>']]
        else:
            amino_seq = [self.vocab_amino['<bos>']] + [self.vocab_amino[amino] for amino in list(seq)] + [self.vocab_amino['<eos>']]

        return amino_seq

    def convert_to_dna(self, seq, seq1_species=None, seq2_species=None, add_species=True):
        if add_species:
            seq1_species_index = self.vocab_dna[seq1_species]
            seq2_species_index = self.vocab_dna[seq2_species]
            dna_seq = [self.vocab_dna['<bos>']] * 3 + [seq1_species_index] * 3 + [seq2_species_index] * 3 + [self.vocab_dna[dna] for dna in list(seq)] + [self.vocab_dna['<eos>']] * 3
        else:
            dna_seq = [self.vocab_dna['<bos>']] * 3 + [self.vocab_dna[dna] for dna in list(seq)] + [self.vocab_dna['<eos>']] * 3
        return dna_seq

    # FASTAファイルからアクセッション番号に対応する配列を読み込む
    def read_fasta(self, fasta_file, accession):
        with open(fasta_file, "r") as f:
            fasta_data = f.read()

        pattern = f">.*{accession}.*\n([A-Z\n]+)"
        match = re.search(pattern, fasta_data)
        if match is not None:
            seq = match.group(1).replace("\n", "")
        elif accession == "empty":
            seq = ""
        else:
            print(f"No match found for pattern {pattern} in data: {fasta_file}")
            seq = None

        return seq

    # 配列ペアが有効であるかどうかを判断する
    def is_valid_pair(self, seq1, seq2, gap_rate):
        length1, length2 = len(seq1), len(seq2)
        valid_chars = set("ATGC")

        return (
            length1 <= 2100
            and length2 <= 2100
            # and 0.97 <= (length1 / length2) <= 1.03
            and gap_rate < 0.20
            and length1 % 3 == 0  # length1 が 3 で割り切れる条件を追加
            and length2 % 3 == 0  # length2 が 3 で割り切れる条件を追加
            and set(seq1.upper()) <= valid_chars
            and set(seq2.upper()) <= valid_chars
        )

    def align(self, seq1, seq2):  
        # DNA配列を定義します
        seq1 = Seq(seq1)
        seq2 = Seq(seq2)

        # DNA配列をアミノ酸配列に変換します
        protein_seq1 = seq1.translate()
        protein_seq2 = seq2.translate()

        # BLOSUM62の置換行列をロードします
        blosum62 = substitution_matrices.load("BLOSUM62")

        # PairwiseAlignerオブジェクトを作成します
        aligner = Align.PairwiseAligner()

        # BLOSUM62をスコア行列として設定します
        aligner.substitution_matrix = blosum62

        # ギャップの開始（作成）と延長に対するペナルティを設定します
        # これらの値は具体的な解析に応じて調整する必要があります
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1

        aligner.mode = 'global'

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
            if aa1 == '-' and aa2 != '-': # seq1にギャップがある場合
                # modified_dna_seq1 += seq2[j:j+3]
                j += 3
            elif aa1 != '-' and aa2 == '-': # seq2にギャップがある場合
                i += 3
            else: # ギャップがない場合
                # self.codon_dict[str(seq2[j:j+3])][self.num_dict[str(seq1[i:i+3])]] += 1
                modified_dna_seq1 += seq1[i:i+3]
                modified_dna_seq2 += seq2[j:j+3]
                i += 3
                j += 3

        # gapの数を数えます
        gap_count = aligned_seq1.count('-') + aligned_seq2.count('-')

        # gap率を計算します
        gap_rate = gap_count / len(aligned_seq1)
        # アラインメントスコアを計算します
        alignment_score = best_alignment.score
        return gap_rate, alignment_score, modified_dna_seq1, modified_dna_seq2


    def len_vocab_input(self):
        return len(self.vocab)

    def len_vocab_pro_input(self):
        return len(self.vocab_amino)

    def len_vocab_dna_input(self):
        return len(self.vocab_dna)

    def len_vocab_target(self):
        return len(self.vocab_target)

    def len_vocab_target_amino(self):
        return len(self.vocab_target_amino)

    def len_vocab_target_dna(self):
        return len(self.vocab_target_dna)

    def split_groups(self, test_ratio=0.1):
        unique_ortholog_groups = list(self.ortholog_groups)
        unique_ortholog_groups = sorted(unique_ortholog_groups)
        random.seed(42)
        random.shuffle(unique_ortholog_groups)
        
        num_test_groups = int(len(unique_ortholog_groups) * test_ratio)
        self.test_groups = set(unique_ortholog_groups[:num_test_groups])
        self.train_groups = set(unique_ortholog_groups[num_test_groups:])
        

    def split_dataset(self,):
        train_data = [entry for entry in self.data if entry[0] not in self.test_groups]
        test_data = [entry for entry in self.data if entry[0] in self.test_groups]        
        return train_data, test_data

    # 指定されたインデックスのデータを返す
    def __getitem__(self, index):
        return self.data[index]


    # データセットのサイズを返す
    def __len__(self):
        return len(self.data)
