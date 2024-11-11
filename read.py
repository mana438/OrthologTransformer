import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
import random
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio import SeqIO
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
    def __init__(self, OMA_species, json_path):
        self.data = []
        self.ortholog_groups = set()
        # すべての菌種名のリストを取得
        with open(OMA_species, 'r') as file:
            species_names = file.read().splitlines()
        
        # すべてのコドンのリスト
        codons = [f"{x}{y}{z}" for x in "ACGT" for y in "ACGT" for z in "ACGT"]
        gap = ["---"]

        # 特殊文字
        spc = ['<pad>', '<bos>', '<eos>']

        # コドンと菌種名の両方を含むトークンのリストを作成
        tokens =  spc + codons + gap + species_names
        self.vocab = Vocab(tokens, json_path)
        self.vocab_target = spc + codons + gap


    def load_data(self, ortholog_files, reverse):
        dataset = []
        # オルソログ関係ファイルを1つずつ読み込む
        for ortholog_file in ortholog_files:
            # 正規表現を用いてグループ番号と菌種名を抽出
            match = re.search(r"group(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_\d+\.fasta", ortholog_file)

            if match:
                group_number = match.group(1)
                species_name_1 = match.group(2)
                species_name_2 = match.group(3)
            else:
                print("No match found")

            sequences = []
            # fastaファイルを読み込み、各配列を辞書に格納
            for seq_record in SeqIO.parse(ortholog_file, "fasta"):
                sequences.append(str(seq_record.seq))

            # ペアが有効である場合、データに追加する
            if self.is_valid_pair(sequences[0], sequences[1]) or group_number == "9999999":
                seq1_codons = self.convert_to_codons(sequences[0], species_name_1)                            
                seq2_codons = self.convert_to_codons(sequences[1], species_name_2)                            
                dataset.append((group_number, seq1_codons, seq2_codons))
                if reverse:
                    dataset.append((group_number, seq2_codons, seq1_codons))
        return dataset


    def convert_to_codons(self, seq, seq_species):
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        seq_species_index = self.vocab[seq_species]
        codon_seq = [seq_species_index] +  [self.vocab['<bos>']] + [self.vocab[codon] for codon in codons] + [self.vocab['<eos>']]
        return codon_seq


    # 配列ペアが有効であるかどうかを判断する
    def is_valid_pair(self, seq1, seq2):
        length1, length2 = len(seq1), len(seq2)
        valid_chars = set("ATGC-")

        return (
            length1 <= 2100
            and length2 <= 2100
            and length1 % 3 == 0  # length1 が 3 で割り切れる条件を追加
            and length2 % 3 == 0  # length2 が 3 で割り切れる条件を追加
            and set(seq1.upper()) <= valid_chars
            and set(seq2.upper()) <= valid_chars
        )

    def len_vocab_input(self):
        return len(self.vocab)

    def len_vocab_target(self):
        return len(self.vocab_target)

    # 指定されたインデックスのデータを返す
    def __getitem__(self, index):
        return self.data[index]


    # データセットのサイズを返す
    def __len__(self):
        return len(self.data)
