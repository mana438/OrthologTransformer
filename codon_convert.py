import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import autocast, GradScaler

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, count_parameters, CG_ratio
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import time
import sys

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

import schedulefree

args = get_args()

#結果ファイル
result_folder = args.result_folder
# 標準出力先の変更
sys.stdout = open(os.path.join(result_folder, "output.txt"), "a")
# パラメータ出力
with open(os.path.join(result_folder, "args.json"), "a") as f:
    json.dump(vars(args), f, indent=4)

# OMA_speciesファイル
OMA_species = args.OMA_species
# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(OMA_species,"./vocab_OMA.json")

start_time = time.time() 
# もし pickle ファイルが存在すれば、そこから読み込む
if args.pickle_path:
    with open(args.pickle_path, "rb") as f:
        train_dataset = pickle.load(f)
else:
    # 学習のみのオルソログ関係ファイルのリスト
    ortholog_files_train = glob.glob(args.ortholog_files_train.replace("'", ""))
    print("pre: ",len(ortholog_files_train), flush=True)
    train_dataset = dataset.load_data(ortholog_files_train, args.reverse, args.calm)

print(f"data load Time: {time.time() - start_time:.2f} seconds", flush=True)
print("train: ", len(train_dataset), flush=True)


# Count unique DNA sequences
unique_sequences = set()
for _, seq1_codons, seq2_codons, _ in train_dataset:
    # Convert codon indices back to sequence representation for uniqueness check
    seq1 = tuple(seq1_codons[2:-1])  # Skip species index, <bos> and <eos>
    seq2 = tuple(seq2_codons[2:-1])  # Skip species index, <bos> and <eos>
    unique_sequences.add(seq1)
    unique_sequences.add(seq2)
print(f"Number of unique DNA sequences: {len(unique_sequences)}", flush=True)    

if args.horovod:
    import horovod.torch as hvd
    hvd.init()
    # horovod用のdataloader
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, collate_fn=custom_collate_fn)
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda", hvd.local_rank())
    print(f"Process {hvd.rank()}: Local rank {hvd.local_rank()}", flush=True)

else:
    hvd = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

# token size
vocab_size_target = dataset.len_vocab_target()
vocab_size_input = dataset.len_vocab_input()

# モデルインスタンス生成
model = CodonTransformer(vocab_size_target, vocab_size_input, args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout).to(device)
# model = torch.compile(model, backend="aot_eager")

# 重みの読み込み
if args.model_input:
    if args.horovod:
        # Horovod 環境なら各プロセスに割り当てられた GPU へマップ
        weights = torch.load(args.model_input, map_location=lambda storage, loc: storage.cuda(hvd.local_rank()))
    else:
        # GPU が1つしかない、または CPU にマップしたいなら以下のように指定
        # weights = torch.load(args.model_input, map_location='cpu')
        weights = torch.load(args.model_input, map_location='cuda:0')
    model.load_state_dict(weights, strict=True)
    
# # 重みの読み込み
# if args.model_input:
#     weights = torch.load(args.model_input)
#     model.load_state_dict(weights, strict=False)
print(f'Total number of parameters: {count_parameters(model)}', flush=True)


# torch.compile()で最適化
# model = torch.compile(model)

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


# 損失関数と最適化アルゴリズム
pad_idx = dataset.vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
# optimizer = optim.NAdam(model.parameters(), lr=args.learning_rate)
# optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
# optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.learning_rate)
# optimizer.train() #schedulefree用

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# GradScalerの初期化
scaler = GradScaler()

if args.horovod:
    # horovod用 optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # horovod用: モデルパラメータとオプティマイザの状態をブロードキャスト
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# 学習
for epoch in range(args.num_epochs):
    start_time = time.time()  # エポック開始時の時間を記録
    try:
        print(epoch, " epoch start", flush=True)
        # 各GPUが扱うデータ数を表示
        if hasattr(train_loader.sampler, 'num_samples') and args.horovod:
            print(f"Process {hvd.rank()} is handling {train_loader.sampler.num_samples} samples.", flush=True)

        model.train()
        
        # トレーニングのループの外で定義します
        train_correct_predictions = 0
        train_total_predictions = 0

        total_loss = 0
        num_batches = 0

        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()  # バッチ処理開始時間
            
            tgt = batch[1].to(device)                
            src = batch[2].to(device)

            dec_ipt = tgt[:, :-1]
            dec_tgt = tgt[:, 1:] #右に1つずらす
            
            # パディングマスクを作成
            src_key_padding_mask = (src == pad_idx).to(device)
            tgt_key_padding_mask = (dec_ipt == pad_idx).to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_codon = model(dec_ipt, src, calm_model(batch[3].to(device), repr_layers=[12])["representations"][12].detach() if args.calm else None, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                # output_codon = model(dec_ipt, src, calm_output)
                output_codon = output_codon.transpose(1, 2) #CrossEntropyLossの時
                loss = criterion(output_codon, dec_tgt)

            # 勾配の初期化
            optimizer.zero_grad()
            
            # 勾配をスケールしてバックプロパゲーション
            scaler.scale(loss).backward()
            
            # Horovodの同期
            if args.horovod:
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)  # スケーリングされた勾配で更新
            else:
                scaler.step(optimizer)
            
            # スケーリングの倍率を更新
            scaler.update()
            
            
            total_loss += loss.item()
            num_batches += 1
            
            # if (epoch + 1) == args.num_epochs:
                # 正解数と全体の予測数を計算
                #コドンの場合
            output_codon = output_codon.transpose(1, 2)
            output_codon = torch.argmax(output_codon, dim=2)
            match_count, all_count = count_nonzero_matches(tgt[:, 1:], output_codon)
            train_correct_predictions += match_count
            train_total_predictions += all_count
                
            batch_time = time.time() - batch_start_time
            # if batch_idx < 5:  # 最初の5バッチの間のみ記録
            #     print(f"Batch {batch_idx + 1}: {batch_time:.4f} seconds", flush=True)

        # if (epoch + 1) == args.num_epochs:
        train_accuracy = train_correct_predictions / train_total_predictions
        print(f"Training Accuracy: {train_accuracy:.4f}", flush=True)
        
        print(f"Epoch {epoch + 1} Time: {time.time() - start_time:.2f} seconds, Avg Batch Time: {(time.time() - start_time) / num_batches:.2f} seconds", flush=True)

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}", flush=True)
        # モデルの state_dict を保存
        torch.save(model.state_dict(), os.path.join(result_folder, os.path.basename("model_weights.pth")))
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb}", flush=True)
        print(f"Skipping this epoch and moving to next epoch.", flush=True)
        continue