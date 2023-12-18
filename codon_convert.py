import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
from torchtext.data.metrics import bleu_score
import torch.nn.functional as F

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, gumbel_softmax, soft_align, count_parameters, CG_ratio
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import sys
from concurrent.futures import ThreadPoolExecutor

from Bio.Seq import Seq

import traceback
import random

import json
import argparse
import glob
from arg_parser import get_args

args = get_args()
if args.horovod:
    import horovod.torch as hvd
    hvd.init()
else:
    hvd = None

# 結果出力フォルダー
import datetime
result_folder = os.path.join("/home/aca10223gf/workplace/job_results/", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(result_folder, exist_ok=True)

# パラメータ出力
with open(os.path.join(result_folder, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

# 標準出力先の変更
sys.stdout = open(os.path.join(result_folder, "output.txt"), "w")

# テスト対象のオルソログ関係ファイルのリスト
ortholog_files_train_test = glob.glob(args.ortholog_files_train_test)

if args.train:
    # 学習のみのオルソログ関係ファイルのリスト
    ortholog_files_train = glob.glob(args.ortholog_files_train.replace("'", ""))

# FASTAファイルが格納されているディレクトリ
fasta_dir = args.fasta_dir

# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(fasta_dir)
# test groupを生成
if args.train:
    dataset.split_groups(ortholog_files_train_test)
else:
    dataset.split_groups(ortholog_files_train_test, 1.0)

# テスト対象のオルソログ関係ファイルを読み込む
dataset.load_data(ortholog_files_train_test, args.edition_fasta, False, args.data_alignment, args.use_gap, args.gap_open, args.gap_extend, args.gap_ratio)
print("train_test dataset: " + str(len(dataset)), flush=True)


if args.train:
    # 学習のみのオルソログ関係ファイルを追加。ただしtest groupは読み込まない
    dataset.load_data(ortholog_files_train, None, True, args.data_alignment, args.use_gap, args.gap_open, args.gap_extend, args.gap_ratio)

# データセットをトレーニングデータとテストデータに分割する
train_dataset, test_dataset = dataset.split_dataset()

#######
print("train: ", len(train_dataset), flush=True)
print("test: ", len(test_dataset), flush=True)
#########

if args.horovod:
    # horovod用のdataloader
    if args.train:
        train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
    # Test sampler and loader
    test_sampler = distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, collate_fn=custom_collate_fn)
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda", hvd.local_rank())
else:
    # トレーニングデータとテストデータを DataLoader でラップする
    if args.train:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

alignment = alignment(dataset.vocab)

# token size
vocab_size_target = dataset.len_vocab_target()
vocab_size_input = dataset.len_vocab_input()

# モデルインスタンス生成
model = CodonTransformer(vocab_size_target, vocab_size_input, args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout).to(device)

# 重みの読み込み
if args.model_input:
    weights = torch.load(args.model_input)
    model.load_state_dict(weights, strict=False)


print(f'Total number of parameters: {count_parameters(model)}', flush=True)

# 損失関数と最適化アルゴリズム
# criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


if args.horovod:
    # horovod用 optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # horovod用: モデルパラメータとオプティマイザの状態をブロードキャスト
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# 学習
if args.train:
    for epoch in range(args.num_epochs):
        try:
            print(epoch, " epoch start", flush=True)
            model.train()
            # トレーニングのループの外で定義します
            train_correct_predictions = 0
            train_total_predictions = 0

            total_loss = 0
            num_batches = 0
            total_soft_align_loss = 0

            for batch in train_loader:
                tgt = batch[1].to(device)                
                src = batch[2].to(device)

                dec_ipt = tgt[:, :-1]
                dec_tgt = tgt[:, 1:] #右に1つずらす
                
                optimizer.zero_grad()
                output_codon = model(dec_ipt, src, args.memory_mask)
                
                output_codon = output_codon.transpose(1, 2) #CrossEntropyLossの時
                loss_codon = criterion(output_codon, dec_tgt)
                loss = loss_codon
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
        
                if (epoch + 1) == args.num_epochs:
                    # 正解数と全体の予測数を計算
                    #コドンの場合
                    output_codon = output_codon.transpose(1, 2)
                    output_codon = torch.argmax(output_codon, dim=2)
                    match_count, all_count = count_nonzero_matches(tgt[:, 1:], output_codon)
                    train_correct_predictions += match_count
                    train_total_predictions += all_count

            if (epoch + 1) == args.num_epochs:
                # 各エポックの最後に、トレーニングの分類予測精度を表示します。
                train_accuracy = train_correct_predictions / train_total_predictions
                print(f"Training Accuracy: {train_accuracy:.4f}", flush=True)
            
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

# 評価
model.eval()
total_alignment_score = 0.0

source_sequences = []
target_sequences = []
predicted_sequences = []

with torch.no_grad():
    # for batch in test_loader:
    #     tgt = batch[1].to(device)  
    #     src = batch[2].to(device)

    #     dec_ipt = tgt[:, :-1]
    #     dec_tgt = tgt[:, 1:] #右に1つずらす

    #     output_codon = model(dec_ipt, src, args.memory_mask)
    #     output_codon = output_codon.transpose(1, 2)
        
    #     # codon rank
    #     # 各位置での正解ラベル（dec_tgt）がoutput_codonで上から何番目に位置するかを調べる
    #     sorted_scores, sorted_indices = output_codon.sort(dim=1, descending=True)
    #     sorted_indices = sorted_indices.transpose(1, 2)  # [batch_size, sequence_length, num_classes]

    #     # dec_tgtを[batch_size, sequence_length, 1]に拡張
    #     expanded_dec_tgt = dec_tgt.unsqueeze(2)  # [batch_size, sequence_length, 1]

    #     # 正解ラベルのランク（位置）を調べる
    #     ranks = (sorted_indices == expanded_dec_tgt).nonzero(as_tuple=True)[2] + 1  # 1-based rank
    #     ranks = ranks.reshape(dec_tgt.shape)  # [batch_size, sequence_length]
    #     print("codon Ranks:", ranks)


    for batch in test_loader:
        tgt = batch[1].to(device)
        src = batch[2].to(device)

        # dec_ipt = torch.tensor([[dataset.vocab['<bos>']]] * len(src), dtype=torch.long, device=device)
        dec_ipt = tgt[:, :2]

        if args.edition_fasta:
            dec_ipt = tgt[:,:-1]

        src_ite = src
        for l in range(30):
            print("eopch ", l, flush=True)
            dec_ipt = tgt[:, :2]
            for i in range(700):
                output_codon_logits = model(dec_ipt, src_ite, args.memory_mask)
                output_codon_probs = F.softmax(output_codon_logits, dim=2)
                output_codon = torch.argmax(output_codon_probs, dim=2)

                # 確率が0.9を上回るかどうかをチェック
                max_probs, max_indices = output_codon_probs.max(dim=2)
                adopt_indices = max_probs > 0.97        
                # 予測されたcodonトークンをデコーダの入力に追加
                try:
                    next_item = torch.where(adopt_indices[:, -1].unsqueeze(1), output_codon[:, -1].unsqueeze(1), src_ite[:, dec_ipt.size(1)].unsqueeze(1))
                except Exception as e:
                    print(f"An error occurred: {e}")
                    # エラーが発生した場合の代替処理
                    next_item = output_codon[:, -1].unsqueeze(1)

                # 最も確率の高いcodonトークンを取得
                # next_item = output_codon[:, -1].unsqueeze(1)

                # 予測されたcodonトークンをデコーダの入力に追加
                dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

                # 文末を表すトークンが出力されたら終了
                # 各行に'<eos>'が含まれているか否かを判定
                end_token_count = (dec_ipt == dataset.vocab['<eos>']).any(dim=1).sum().item()
                if end_token_count == len(src):
                    # src_ite = dec_ipt.detach()
                    src_ite = torch.cat((src[:, :2], dec_ipt[:, 2:]), dim=1)                        
                    break

        source_sequences += alignment.extract_sequences(src[:,1:])
        target_sequences += alignment.extract_sequences(tgt[:,1:])
        predicted_sequences += alignment.extract_sequences(dec_ipt[:,1:])

    # ギャップを取り除く
    predicted_sequences = [[item for item in row if item != dataset.vocab['---']] for row in predicted_sequences]
    
    if args.train:
        if args.horovod:            
            source_sequences, target_sequences , predicted_sequences = allreduce(source_sequences, target_sequences, predicted_sequences, dataset.vocab)
            # if hvd.rank() == 0:  # ここでランク0のノードのみが実行
        plot_obj = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
        plot_obj.savefig(os.path.join(result_folder, "align.png"))

    predicted_sequences = [[dataset.vocab.index_to_token[codon] for codon in sequence] for sequence in predicted_sequences]
    predicted_sequences_all = [''.join(sequence) for sequence in predicted_sequences]
    print("GC ratio " +  str(CG_ratio(predicted_sequences_all)))
    print('\n'.join(predicted_sequences_all))

    if args.mcts:
        mcts(model, test_loader, dataset.vocab, device, args.edition_fasta, predicted_sequences, args.memory_mask, result_folder)



    