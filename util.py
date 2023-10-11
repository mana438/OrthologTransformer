from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import torch.nn as nn
from Bio import pairwise2
from Bio.Seq import Seq
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

            extracted_sequence = row[start_idx:end_idx].tolist()
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
        aligned_seq1, aligned_seq2, score, _, _ = alignments[0]
        aligned_seq1 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq1) ])
        aligned_seq2 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq2)])
        

        return Seq(aligned_seq1), Seq(aligned_seq2), score

    def chunks(self, lst, chunk_size):
        """リストを指定されたチャンクサイズで分割するジェネレータ"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def plot_alignment_scores(self, src_seqs, tgt_seqs, pred_seqs, xlabel="Alignment score of source and target codons", ylabel="Alignment score of target and predicted codons"):
        src_tgt_scores = []
        tgt_pred_scores = []
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
        # filename = "/home/aca10223gf/workplace/job_results/align.png"
        # counter = 1
        # while os.path.exists(filename):
        #     # ファイル名の拡張子の前にカウンタを追加する
        #     filename = f"/home/aca10223gf/workplace/job_results/align_png/align_{counter}.png"
        #     counter += 1
        # # 画像をファイルに保存する
        # plt.savefig(filename)
        return plt
    
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

# def replace_elements_with_probability(lst, value, probability):
#     for i, element in enumerate(lst):
#         if random.random() < probability:
#             lst[i] = value
#     return lst

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

# ガンベルノイズを生成する関数
# shape: 出力テンソルの形状
# device: 使用するデバイス（CPUまたはGPU）
# eps: 数値安定性のための微小値
def gumbel_noise(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)  # [0, 1)の一様分布からサンプリング
    return -torch.log(-torch.log(U + eps) + eps)  # ガンベル分布に従うノイズ

# ガンベルソフトマックス関数
# logits: 元の確率の対数値（ロジット）
# tau: 温度パラメータ（低い値で確定的な出力、高い値で確率的な出力）
# hard: 出力をハードにするかどうか（つまり、one-hotにするかどうか）
def gumbel_softmax(logits, tau=1.0, hard=False):
    shape = logits.size()  # 元のテンソルの形状
    device = logits.device  # 使用するデバイス

    gumbel = gumbel_noise(shape, device)  # ガンベルノイズを生成
    y = logits + gumbel  # ノイズをロジットに加える

    y_soft = torch.nn.functional.softmax(y / tau, dim=-1)  # ソフトマックスを適用

    # hardがTrueの場合は、出力をほぼone-hotにする
    if hard:
        _, max_idx = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, max_idx.unsqueeze(-1), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft 
    return y  # 結果を返す

def soft_align(encoder_out, decoder_out):
    # Step 1: Expand dimensions
    # shape -> (batch_size, encoder_seq_len, 1, hidden_size)
    # shape -> (batch_size, 1, decoder_seq_len, hidden_size)
    expanded_encoder_out = encoder_out.unsqueeze(2)
    expanded_decoder_out = decoder_out.unsqueeze(1)

    # Step 2: Repeat Tensors
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len, hidden_size)
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len, hidden_size)
    repeated_encoder_out = expanded_encoder_out.repeat(1, 1, decoder_out.size()[1], 1)
    repeated_decoder_out = expanded_decoder_out.repeat(1, encoder_out.size()[1], 1, 1)

    # Step 3: Compute cosine similarity
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len)
    cosine_similarity = nn.CosineSimilarity(dim=3, eps=1e-6)
    match_score = cosine_similarity(repeated_encoder_out, repeated_decoder_out)
    s = match_score
    # Step 4: Soft alignment
    # Calculate softmax along the encoder sequence (axis=1)
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len)
    a = F.softmax(s, dim=1)

    # Calculate softmax along the decoder sequence (axis=2)
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len)
    b = F.softmax(s, dim=2)

    # Combine the alignments
    # shape -> (batch_size, encoder_seq_len, decoder_seq_len)
    c = a + b - a * b

    # Calculate the final aligned score (optional)
    # shape -> (batch_size,)
    aligned_score = (c * s).sum(dim=(1,2)) / c.sum(dim=(1,2))
    shifted_score = (aligned_score + 1)/2
    distance = (1 / (shifted_score + 1e-6)) -1
    return distance

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def randomize_nested_list(nested_list, ratio):

    randomized_list = copy.deepcopy(nested_list)
    print(randomized_list)
    for i in range(len(randomized_list)):
        ratio_mod = ratio + (random.random()-0.5)/10
        for j in range(len(randomized_list[i])):
            if random.random() < ratio_mod:
                randomized_list[i][j] = random.randint(5, 6)
                
    return randomized_list

def CG_ratio(sequences):
    gc_contents = [ seq.count('G') + seq.count('C') for seq in sequences]
    return sum(gc_contents) / len(gc_contents)