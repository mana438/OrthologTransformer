import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, count_parameters, CG_ratio, save_with_unique_filename
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import time
import sys

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import traceback
import random

import json
import argparse
import glob
from arg_parser import get_args
import pickle


from calm.sequence import CodonSequence  # CodonSequenceクラス
from calm.alphabet import Alphabet  # アルファベット定義
from calm.model import ProteinBertModel
from calm.pretrained import ARGS  # 事前に定義されたARGSを使用


args = get_args()

result_folder = args.result_folder
# 標準出力先の変更
sys.stdout = open(os.path.join(result_folder, "output.txt"), "a") 
# パラメータ出力
with open(os.path.join(result_folder, "args_test.json"), "w") as f:
    json.dump(vars(args), f, indent=4)


# OMA_speciesファイル
OMA_species = args.OMA_species
# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(OMA_species,"./vocab_OMA.json")
# テスト対象のオルソログ関係ファイルのリスト
ortholog_files_test = glob.glob(args.ortholog_files_test.replace("'", ""))
test_dataset = dataset.load_data(ortholog_files_test, False, args.calm)
print("test: ", len(test_dataset), flush=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPUが利用可能な場合は 'cuda:0' にマップ、そうでない場合は CPU にマップ
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
# token size
vocab_size_target = dataset.len_vocab_target()
vocab_size_input = dataset.len_vocab_input()

# モデルインスタンス生成
model = CodonTransformer(vocab_size_target, vocab_size_input, args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout).to(device)
# model = torch.compile(model, backend="aot_eager")

# 重みの読み込み
if args.model_input:
    # モデルをロード
    weights = torch.load(args.model_input, map_location=device)
    model.load_state_dict(weights, strict=True)

print(f'Total number of parameters: {count_parameters(model)}', flush=True)


if args.calm:
    # calm適用
    weights_file = os.path.join('/home/4/ux03574/workplace/CodonTransfer/calm/calm_weights/calm_weights.ckpt')
    # if not os.path.exists(weights_file):
    #     print('Downloading model weights...')
    #     os.makedirs(model_folder, exist_ok=True)
    #     url = 'http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl'
    #     with open(weights_file, 'wb') as handle:
    #         handle.write(requests.get(url).content)
                
    # GPUの使用を確認
    alphabet = Alphabet.from_architecture("CodonModel")
    batch_converter = alphabet.get_batch_converter()

    # モデルの初期化
    calm_model = ProteinBertModel(ARGS, alphabet).to(device)
    with open(weights_file, 'rb') as handle:
        state_dict = pickle.load(handle)
        calm_model.load_state_dict(state_dict)
        
    # calm_model = torch.compile(calm_model)
    
    # calm_modelのパラメータをフリーズ
    for param in calm_model.parameters():
        param.requires_grad = False
        
    if args.horovod:
        hvd.broadcast_parameters(calm_model.state_dict(), root_rank=0)

# torch.compile()で最適化
# model = torch.compile(model, backend="aot_eager")
# model = torch.compile(model)

# 評価
model.eval()
total_alignment_score = 0.0
alignment = alignment(dataset.vocab)
pad_idx = dataset.vocab['<pad>']

source_sequences = []
target_sequences = []
predicted_sequences = []

with torch.no_grad():
    for batch in test_loader:
        tgt = batch[1].to(device)
        src = batch[2].to(device)

        # dec_ipt = torch.tensor([[dataset.vocab['<bos>']]] * len(src), dtype=torch.long, device=device)
        dec_ipt = tgt[:, :2]

        if args.edition_fasta:
            dec_ipt = tgt[:,:-1]
        
        src_key_padding_mask = (src == pad_idx).to(device)
        # tgt_key_padding_mask = (dec_ipt == pad_idx).to(torch.bfloat16).to(device)

        for i in range(700):
            with autocast(device_type='cuda', dtype=torch.bfloat16):  # 評価でもbf16演算を有効化
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_codon_logits = model(dec_ipt, src, calm_model(batch[3].to(device), repr_layers=[12])["representations"][12].detach() if args.calm else None, src_key_padding_mask=src_key_padding_mask)
                output_codon_probs = F.softmax(output_codon_logits, dim=2)
                output_codon = torch.argmax(output_codon_probs, dim=2)

            # 最も確率の高いcodonトークンを取得
            next_item = output_codon[:, -1].unsqueeze(1)

            # 予測されたcodonトークンをデコーダの入力に追加
            dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

            # 文末を表すトークンが出力されたら終了
            # 各行に'<eos>'が含まれているか否かを判定
            end_token_count = (dec_ipt == dataset.vocab['<eos>']).any(dim=1).sum().item()
            if end_token_count == len(src):
                break

        source_sequences += alignment.extract_sequences(src[:,1:])
        target_sequences += alignment.extract_sequences(tgt[:,1:])
        predicted_sequences += alignment.extract_sequences(dec_ipt[:,1:])

    
    # ギャップを取り除く
    predicted_sequences = [[item for item in row if item != dataset.vocab['---']] for row in predicted_sequences]
    plot_obj, score_ratio = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
    save_with_unique_filename(result_folder, "align.png", plot_obj)
    
    predicted_sequences_all =  [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in predicted_sequences]
    source_sequences_dna = [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in source_sequences]
    target_sequences_dna = [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in target_sequences]
    predicted_protein_sequences_all = [str(Seq(dna).translate(to_stop=True)) for dna in predicted_sequences_all]
    source_protein_sequences = [str(Seq(dna).translate(to_stop=True)) for dna in source_sequences_dna]
    target_protein_sequences = [str(Seq(dna).translate(to_stop=True)) for dna in target_sequences_dna]

    # アミノ酸散布図を作成
    plot_obj_aa, score_ratio_aa = alignment.plot_alignment_scores_aa(source_protein_sequences, target_protein_sequences, predicted_protein_sequences_all)
    save_with_unique_filename(result_folder, "align_aa.png", plot_obj_aa)


    print("GC ratio " +  str(CG_ratio(predicted_sequences_all)))
    # print('\n'.join(predicted_sequences_all))
    
    # ファイルパスを指定
    # ユニークなファイルパスを生成
    fasta_output_path = os.path.join(result_folder, "predicted_sequences.fasta")
    counter = 1
    while os.path.exists(fasta_output_path):
        fasta_output_path = os.path.join(result_folder, f"predicted_sequences_{counter}.fasta")
        counter += 1
    # FASTA形式で保存
    SeqIO.write([SeqRecord(Seq(seq), id=f"seq_{i+1}", description="") for i, seq in enumerate(predicted_sequences_all)], fasta_output_path, "fasta")
    
    print(f"Predicted sequences saved to {fasta_output_path}")

    if args.mcts:
        mcts(model, test_loader, dataset.vocab, device, args.edition_fasta, predicted_sequences, args.memory_mask, result_folder)
