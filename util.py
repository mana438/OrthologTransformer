from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import torch.nn as nn
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import os
import random
import statistics
import numpy as np
import json
import copy

class alignment:
    def  __init__(self, vocab):
        self.vocab = vocab
        # 数値 3 から 66 を適当な 1 文字にマッピングする辞書
        self.num_to_char = {}
        self.char_to_num = {}
        start_ascii = 33  # '!' の ASCII コード
        for i in range(3, 67):
            char = chr(start_ascii + i - 3)
            if char == '-':
                start_ascii += 1
                char = chr(start_ascii + i - 3)
            self.num_to_char[i] = char
            self.char_to_num[char] = i

    # 1と2に挟まれた配列を抽出    
    def extract_sequences(self, tensor):
        result = []
        for row in tensor:
            # start_idx = (row == 1).nonzero(as_tuple=True)[0][0] + 1
            one_indices = (row == 1).nonzero(as_tuple=True)[0]
            # 1が存在しない場合、行の最初から抽出を開始します
            start_idx = one_indices[0] + 1 if one_indices.size(0) > 0 else 0
        
            end_indices_2 = (row == 2).nonzero(as_tuple=True)[0]
            end_indices_0 = (row == 0).nonzero(as_tuple=True)[0]

            if end_indices_2.size(0) == 0:
                end_idx_2 = row.size(0)
            else:
                end_idx_2 = end_indices_2[0]

            if end_indices_0.size(0) == 0:
                end_idx_0 = row.size(0)
            else:
                end_idx_0 = end_indices_0[0]

            end_idx = min(end_idx_2, end_idx_0)

            #extracted_sequence = row[start_idx:end_idx].tolist()
            # **抽出された配列から0、1、2を取り除く**
            extracted_sequence = [num for num in row[start_idx:end_idx].tolist() if num not in [0, 1, 2]]
            result.append(extracted_sequence)
        return result


    # 数値リストを文字列に変換する関数
    def convert_num_list_to_string(self, num_list, num_to_char):
        return ''.join([self.num_to_char[num] for num in num_list])


    # シーケンスをアラインメントする関数
    def align_sequences(self, seq_pair):
        seq1, seq2 = seq_pair
        seq1 = ''.join([self.num_to_char[num] for num in seq1])
        seq2 = ''.join([self.num_to_char[num] for num in seq2])  
          
        alignments = pairwise2.align.globalxx(Seq(seq1), Seq(seq2))

        if not alignments:
            print(f"Warning: No alignments found for sequences: {seq1}, {seq2}")
            return seq1, seq2, 0

        aligned_seq1, aligned_seq2, score, _, _ = alignments[0]
        aligned_seq1 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq1) ])
        aligned_seq2 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq2)])
        

        return Seq(aligned_seq1), Seq(aligned_seq2), score

    # align structure
    def align_structures(self, seq_pair):
        seq1, seq2 = seq_pair
        seq1 = ''.join([self.vocab.index_to_token[num] for num in seq1])
        seq2 = ''.join([self.vocab.index_to_token[num] for num in seq2])  
        
    def predict_structure(self, dna_sequence):

        # Biopythonを用いてDNA配列をアミノ酸配列に変換
        dna_seq = Seq(dna_sequence)
        amino_acid_seq = dna_seq.translate()


        # アミノ酸配列をFASTA形式で保存
        fasta_file_path = '../job_result/query.fasta'
        record = SeqRecord(amino_acid_seq, id="query_sequence", description="Translated sequence")
        os.makedirs(os.path.dirname(fasta_file_path), exist_ok=True)  # ディレクトリが存在しない場合に作成
        with open(fasta_file_path, 'w') as fasta_file:
            SeqIO.write(record, fasta_file, "fasta")

        # 実行スクリプトのパス
        alphafold_script_path = '../alphafold/run_alphafold.sh'
        # シェルコマンドをディレクトリ移動せずに実行
        result = subprocess.run([
            alphafold_script_path,
            '-a', '0,1,2,3',
            '-d', os.getenv('ALPHAFOLD_DATA_DIR'),
            '-o', '../dummy_test/',
            '-m', 'model_1',
            '-f', fasta_file_path,
            '-t', '2020-05-14'
        ], capture_output=True, text=True)

        # 実行結果を表示
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)




    def chunks(self, lst, chunk_size):
        """リストを指定されたチャンクサイズで分割するジェネレータ"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def plot_alignment_scores(self, src_seqs, tgt_seqs, pred_seqs, xlabel="Alignment score of source and target codons", ylabel="Alignment score of target and predicted codons"):
        src_tgt_scores = []
        tgt_pred_scores = []
        score_ratio = []
        # リストの各要素から10個ずつ取り出してタプルにし、それらのタプルのリストを作成
        batch_num = 30
        src_seqs = list(self.chunks(src_seqs, batch_num))
        tgt_seqs = list(self.chunks(tgt_seqs, batch_num))
        pred_seqs = list(self.chunks(pred_seqs, batch_num))

        for src_seq, tgt_seq, pred_seq in zip(src_seqs, tgt_seqs, pred_seqs):
            # print(tgt_seq)
            # source_sequences, target_sequencesをペアワイズアラインメント
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_src_tgt = list(executor.map(self.align_sequences, zip(src_seq, tgt_seq)))
            aligned_src, aligned_tgt, src_tgt_score = zip(*results_src_tgt)

            # target_sequences,predicted_sequencesをペアワイズアラインメント
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_tgt_pred = list(executor.map(self.align_sequences, zip(tgt_seq, pred_seq)))
            aligned_tgt, aligned_pred, tgt_pred_score = zip(*results_tgt_pred)
            # self.print_alignment(aligned_tgt, aligned_pred, tgt_pred_score) #アラインメントを出力

            # アラインメントスコアを配列の長さで補正
            src_tgt_score_corrected = [score / len(tgt) for score, tgt in zip(src_tgt_score, tgt_seq)]
            tgt_pred_score_corrected = [score / len(tgt) for score, tgt in zip(tgt_pred_score, tgt_seq)]
            

            score_ratio += [pred_score / src_score if src_score != 0 else 0 for pred_score, src_score in zip(tgt_pred_score_corrected, src_tgt_score_corrected)]

            src_tgt_scores += src_tgt_score_corrected
            tgt_pred_scores += tgt_pred_score_corrected
        
        plt.figure()
        # 散布図をプロット
        plt.scatter(src_tgt_scores, tgt_pred_scores)

        src_tgt_mean = np.mean(src_tgt_scores)
        tgt_pred_mean = np.mean(tgt_pred_scores)

        ax = plt.gca()  # 現在のAxesオブジェクトを取得

        # 右下にsrc_tgt_scoresの平均値を表示
        plt.text(0.95, 0.05, f'src-tgt Ave: {src_tgt_mean:.3f}', 
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=12)

        # 左上にtgt_pred_scoresの平均値を表示
        plt.text(0.05, 0.95, f'tgt-pred Ave: {tgt_pred_mean:.3f}', 
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, fontsize=12)
        
        # x軸とy軸の最小値と最大値を計算
        min_value = min(min(src_tgt_scores), min(tgt_pred_scores))
        max_value = max(max(src_tgt_scores), max(tgt_pred_scores))

        # min_value = 0.2
        # max_value = 0.6

        
        # x=y の直線をプロット
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt, score_ratio
    
    def print_alignment(self, aligned_seq1, aligned_seq2, score):
        # アラインメントされたシーケンスを表示
        for aligned_seq1, aligned_seq2, score in zip(aligned_sequences1, aligned_sequences2, scores):
            print("score: " + str(score))
            print("target sequences")
            print(aligned_seq1)
            print("predicted sequences")
            print(aligned_seq2)
            print("-------")
            total_alignment_score += score
        num_samples += len(src)


def custom_collate_fn(batch):
    ortholog_groups, targets, inputs = [], [], []

    for ortholog_group, tgt, inp in batch:
        ortholog_groups.append(ortholog_group)
        targets.append(torch.tensor(tgt))
        
        inputs.append(torch.tensor(inp))

    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    return ortholog_groups, targets_padded, inputs_padded

def count_nonzero_matches(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "The tensors are not the same shape"

    # テンソル同士が一致している部分をTrue、それ以外をFalseとするBoolテンソルを作成
    matches = tensor1 == tensor2

    # 0同士の一致は無視するため、非ゼロ要素のみを対象とする
    nonzero_matches = matches & (tensor1 != 0) & (tensor2 != 0)

    # 非ゼロ要素の一致数を計算
    nonzero_match_count = nonzero_matches.sum().item()

    # テンソル1とテンソル2の非ゼロ要素ペアの数を計算
    nonzero_element_pairs = (tensor1 != 0) | (tensor2 != 0)
    nonzero_element_pair_count = nonzero_element_pairs.sum().item()

    return nonzero_match_count, nonzero_element_pair_count
    
#　パラメータを読み込む
def load_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params


def check_condition(args, hvd):
    if args.horovod:
        if hvd.rank() == 0:
            return True
        else:
            return False
    else:
        return True

def allreduce(source_sequences, target_sequences, predicted_sequences, vocab):
    import horovod.torch as hvd
    # 最大の長さに合わせて、シーケンスをパディング
    source_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in source_sequences]
    target_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in target_sequences]
    predicted_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in predicted_sequences]

    # 既存のhvd.allgatherコード
    all_source_sequences = hvd.allgather(torch.tensor(source_sequences))
    all_target_sequences = hvd.allgather(torch.tensor(target_sequences))
    all_predicted_sequences = hvd.allgather(torch.tensor(predicted_sequences))

     # パディング部分を取り除く
    all_source_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_source_sequences]
    all_target_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_target_sequences]
    all_predicted_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_predicted_sequences]
    return  all_source_sequences, all_target_sequences , all_predicted_sequences

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def CG_ratio(sequences):
    gc_contents = [ (seq.count('G') + seq.count('C')) / len(seq) for seq in sequences]
    return sum(gc_contents) / len(gc_contents)
