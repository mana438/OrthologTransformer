import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
import random
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
import time
import json

class Vocab:
    def __init__(self, tokens, json_path):
        self.json_path = json_path
        self.token_to_index = {}
        self.index_to_token = {}
        self.load_json()  # JSONファイルからデータを読み込む
        self.build_vocab(tokens)

    def load_json(self):
        if os.path.exists(self.json_path) and os.path.getsize(self.json_path) > 0:
            try:
                with open(self.json_path, 'r') as file:
                    data = json.load(file)
                    self.token_to_index = data.get('token_to_index', {})
                    self.index_to_token = {int(key): val for key, val in data.get('index_to_token', {}).items()}
            except json.JSONDecodeError:
                # JSON ファイルが空または不正な場合は、初期化する
                self.token_to_index = {}
                self.index_to_token = {}
        else:
            # ファイルが存在しない場合は、初期化する
            self.token_to_index = {}
            self.index_to_token = {}


    def save_json(self):
        data = {
            'token_to_index': self.token_to_index,
            'index_to_token': {str(key): val for key, val in self.index_to_token.items()}
        }
        with open(self.json_path, 'w') as file:
            json.dump(data, file)

    def build_vocab(self, tokens):
        for token in tokens:
            if token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token
                self.save_json()  # 新しいトークンを追加するたびにJSONファイルを更新


    def __len__(self):
        return len(self.token_to_index)
    
    def __getitem__(self, token):
        return self.token_to_index[token]

class OrthologDataset(Dataset):
    def __init__(self, fasta_dir, json_path = "/home/aca10223gf/workplace/mtgenome/vocab.json"):
        self.data = []
        self.ortholog_groups = set()
        self.fasta_dir = fasta_dir
        # すべての菌種名のリストを取得
        species_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(self.fasta_dir) if file_name.endswith(".fasta")]
        # すべてのコドンのリスト
        codons = [f"{x}{y}{z}" for x in "ACGT" for y in "ACGT" for z in "ACGT"]
        gap = ["---"]

        # 特殊文字
        spc = ['<pad>', '<bos>', '<eos>']

        # コドンと菌種名の両方を含むトークンのリストを作成
        tokens =  spc + codons + gap + species_names
        self.vocab = Vocab(tokens, json_path)
        self.vocab_target = spc + codons + gap


    def load_data(self, ortholog_files, edition_file=None, exclude_test_group=False, data_alignment=False, use_gap=False, gap_open=-10, gap_extend=-0.1, gap_ratio=0.3):
        # オルソログ関係ファイルを1つずつ読み込む
        for ortholog_file in ortholog_files:
            with open(ortholog_file, "r") as f:
                ortholog_data = f.readlines()
            seq1_species = ortholog_data[0].split("\t")[1].strip()
            if edition_file:
                seq1_fasta_file = edition_file
            else:
                seq1_fasta_file = os.path.join(self.fasta_dir, f"{seq1_species}.fasta")
            seq2_species = ortholog_data[0].split("\t")[2].strip()
            seq2_fasta_file = os.path.join(self.fasta_dir, f"{seq2_species}.fasta")  
            
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
                        seq2_seq = self.read_fasta(seq2_fasta_file, seq2_id)
                        if not seq2_seq:
                            continue
                        try:
                            gap_rate, alignment_score, modified_dna_seq1, modified_dna_seq2 = self.align(seq1_seq, seq2_seq, ortholog_group, data_alignment, use_gap, gap_open, gap_extend)
                            seq2_seqs_mod.append(modified_dna_seq2)
                            seq1_seqs_mod.append(modified_dna_seq1)
                            scores.append(alignment_score)
                            gap_rates.append(gap_rate)
                        except ValueError as e:
                            print("value error")

                    if seq1_seq == None or not bool(seq2_seqs_mod):
                        continue

                    # 入力配列の中で最も出力配列と類似度を高いものを選定    
                    seq2_seq_mod = seq2_seqs_mod[scores.index(max(scores))]
                    seq1_seq_mod = seq1_seqs_mod[scores.index(max(scores))]
                    gap_rate = gap_rates[scores.index(max(scores))]

                    # ペアが有効である場合、データに追加する
                    if self.is_valid_pair(seq1_seq_mod, seq2_seq_mod, gap_rate, gap_ratio) or ortholog_group == "OG999999":
                        seq1_codons = self.convert_to_codons(seq1_seq_mod, seq1_species)                            
                        seq2_codons = self.convert_to_codons(seq2_seq_mod, seq2_species)                            
                        self.data.append((ortholog_group, seq1_codons, seq2_codons))


    def convert_to_codons(self, seq, seq_species):
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        seq_species_index = self.vocab[seq_species]
        codon_seq = [seq_species_index] +  [self.vocab['<bos>']] + [self.vocab[codon] for codon in codons] + [self.vocab['<eos>']]
        return codon_seq


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
    def is_valid_pair(self, seq1, seq2, gap_rate, gap_ratio):
        length1, length2 = len(seq1), len(seq2)
        valid_chars = set("ATGC-")

        return (
            length1 <= 2100
            and length2 <= 2100
            # and 0.97 <= (length1 / length2) <= 1.03
            and gap_rate < gap_ratio
            and length1 % 3 == 0  # length1 が 3 で割り切れる条件を追加
            and length2 % 3 == 0  # length2 が 3 で割り切れる条件を追加
            and set(seq1.upper()) <= valid_chars
            and set(seq2.upper()) <= valid_chars
        )

    def align(self, seq1, seq2, ortholog_group, data_alignment, use_gap, gap_open, gap_extend):  
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
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend

        aligner.mode = 'global'

        if len(protein_seq1) == 0 or len(protein_seq2) == 0:
            return 0, 0, seq1, seq2
        else:
            # アミノ酸配列をアラインメントします
            alignments = aligner.align(protein_seq1, protein_seq2)

        # 最良のアラインメントを取得します
        best_alignment = alignments[0]
        aligned_seq1 = str(best_alignment[0])
        aligned_seq2 = str(best_alignment[1])
        # gapの数を数えます
        gap_count = aligned_seq1.count('-') + aligned_seq2.count('-')

        # gap率を計算します
        gap_rate = gap_count / len(aligned_seq1)
        # アラインメントスコアを計算します
        alignment_score = best_alignment.score


        # DNA配列の変更を計算
        modified_dna_seq1 = ""
        modified_dna_seq2 = ""
        if ortholog_group not in self.test_groups and data_alignment:
            i, j = 0, 0
            for aa1, aa2 in zip(best_alignment[0], best_alignment[1]):
                if use_gap:                        
                    if aa1 == '-' : # seq1にギャップがある場合
                        modified_dna_seq1 += "---"
                    else: # ギャップがない場合
                        modified_dna_seq1 += seq1[i:i+3]
                        i += 3
                    if aa2 == '-':
                        modified_dna_seq2 += "---"
                    else:
                        modified_dna_seq2 += seq2[j:j+3]
                        j += 3
                else:
                    if aa1 == '-' and aa2 != '-': # seq1にギャップがある場合
                        j += 3
                    elif aa1 != '-' and aa2 == '-': # seq2にギャップがある場合
                        i += 3
                    else: # ギャップがない場合
                        # self.codon_dict[str(seq2[j:j+3])][self.num_dict[str(seq1[i:i+3])]] += 1
                        modified_dna_seq1 += seq1[i:i+3]
                        modified_dna_seq2 += seq2[j:j+3]
                        i += 3
                        j += 3

            return gap_rate, alignment_score, modified_dna_seq1, modified_dna_seq2

        else:
            return gap_rate, alignment_score, seq1, seq2


    def len_vocab_input(self):
        return len(self.vocab)

    def len_vocab_target(self):
        return len(self.vocab_target)

    def split_groups(self, ortholog_files, test_ratio=0.1):
        for ortholog_file in ortholog_files:
            with open(ortholog_file, "r") as f:
                ortholog_data = f.readlines()
            for line in ortholog_data[1:]:
                ortholog_group, _, _ = line.strip().split("\t")
                self.ortholog_groups.add(ortholog_group)

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
