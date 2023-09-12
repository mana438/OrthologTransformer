import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
from torchtext.data.metrics import bleu_score

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, gumbel_softmax, soft_align
from model import CodonTransformer
from read import OrthologDataset

import os
import sys
from concurrent.futures import ThreadPoolExecutor

from Bio.Seq import Seq

import traceback

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
# テスト対象のオルソログ関係ファイルを読み込む
dataset.load_data(ortholog_files_train_test, False)
print("train_test dataset: " + str(len(dataset)), flush=True)
# test groupを生成
if args.train:
    dataset.split_groups()
else:
    dataset.split_groups(1.0)

if args.train:
    # 学習のみのオルソログ関係ファイルを追加。ただしtest groupは読み込まない
    dataset.load_data(ortholog_files_train, True)

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
vocab_size_target =dataset.len_vocab_target()
vocab_size_target_amino =dataset.len_vocab_target_amino()
vocab_size_target_dna =dataset.len_vocab_target_dna()

vocab_size_input = dataset.len_vocab_input()
vocab_size_input_amino = dataset.len_vocab_pro_input()
vocab_size_input_dna = dataset.len_vocab_dna_input()

# モデルインスタンス生成
model = CodonTransformer(vocab_size_target, vocab_size_target_amino, vocab_size_target_dna, vocab_size_input,  vocab_size_input_amino, vocab_size_input_dna, args.d_model, args.nhead, args.num_layers, args.dim_feedforward, args.dropout).to(device)

# 重みの読み込み
if args.model_input:
    weights = torch.load(args.model_input)
    model.load_state_dict(weights, strict=False)


# 損失関数と最適化アルゴリズム
# criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
criterion = nn.CrossEntropyLoss()
# criterion =  nn.MSELoss()
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
                tgt_protein = batch[2].to(device)
                tgt_dna = batch[3].to(device)
                
                src = batch[4].to(device)
                src_protein = batch[5].to(device)
                src_dna = batch[6].to(device)

                dec_ipt = tgt[:, :-1]
                dec_tgt = tgt[:, 1:] #右に1つずらす
                
                dec_ipt_amino = tgt_protein[:, :-1]
                dec_tgt_amino = tgt_protein[:, 1:] #右に1つずらす

                dec_ipt_dna = tgt_dna[:, :-3]
                dec_tgt_dna = tgt_dna[:, 3:] #右に3つずらす
                
                # dec_tgt = nn.functional.one_hot(dec_tgt, vocab_size_target).to(torch.float32).to(device) #MSELossの時
                optimizer.zero_grad()
                output_codon, output_pro, output_dna, memory, output = model(dec_ipt, dec_ipt_amino, dec_ipt_dna, src, src_protein, src_dna)
                
                output_codon = output_codon.transpose(1, 2) #CrossEntropyLossの時
                loss_codon = criterion(output_codon, dec_tgt)

                output_pro = output_pro.transpose(1, 2) #CrossEntropyLossの時
                loss_pro= criterion(output_pro, dec_tgt_amino)

                output_dna = output_dna.transpose(1, 2) #CrossEntropyLossの時
                loss_dna = criterion(output_dna, dec_tgt_dna)

                # distance = soft_align(memory.detach(), output)
                # if distance.item() < 0.3:
                #     loss = loss_codon
                # else:
                #     loss = loss_codon +  30 * distance

                loss = loss_codon
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_soft_align_loss += distance.item()
                num_batches += 1
        
                if (epoch + 1) == args.num_epochs:
                    # 正解数と全体の予測数を計算
                    #コドンの場合
                    output_codon = output_codon.transpose(1, 2)
                    output_codon = torch.argmax(output_codon, dim=2)
                    match_count, all_count = count_nonzero_matches(tgt[:, 1:], output_codon)
                    train_correct_predictions += match_count
                    train_total_predictions += all_count

                    #DNAの場合
                    # output_dna = output_dna.transpose(1, 2)
                    # output_dna = torch.argmax(output_dna, dim=2)
                    # match_count, all_count = count_nonzero_matches(tgt_dna[:, 3:], output_dna)
                    # train_correct_predictions += match_count
                    # train_total_predictions += all_count
            if (epoch + 1) == args.num_epochs:
                # 各エポックの最後に、トレーニングの分類予測精度を表示します。
                train_accuracy = train_correct_predictions / train_total_predictions
                print(f"Training Accuracy: {train_accuracy:.4f}", flush=True)
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}", flush=True)
            avg_soft_align_loss  = total_soft_align_loss / num_batches
            print(f"Epoch {epoch + 1}/{args.num_epochs}, soft_Loss: {avg_soft_align_loss:.4f}", flush=True)

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
    for batch in test_loader:
        dec_tgt = batch[1].to(device)
        dec_tgt_protein = batch[2].to(device) 
        dec_tgt_dna = batch[3].to(device)    
        src = batch[4].to(device)
        src_protein = batch[5].to(device)
        src_dna = batch[6].to(device)

        dec_ipt = dec_tgt[:, :-1]
        dec_tgt = dec_tgt[:, 1:] #右に1つずらす
        
        dec_ipt_amino = dec_tgt_protein[:, :-1]
        dec_tgt_amino = dec_tgt_protein[:, 1:] #右に1つずらす

        dec_ipt_dna = dec_tgt_dna[:, :-3]
        dec_tgt_dna = dec_tgt_dna[:, 3:] #右に1つずらす

        output_codon, output_pro, output_dna, _, _ = model(dec_ipt, dec_ipt_amino, dec_ipt_dna,  src, src_protein, src_dna)
        output_codon = output_codon.transpose(1, 2)
        output_dna = output_dna.transpose(1, 2)
        
        # codon rank
        # 各位置での正解ラベル（dec_tgt）がoutput_codonで上から何番目に位置するかを調べる
        sorted_scores, sorted_indices = output_codon.sort(dim=1, descending=True)
        sorted_indices = sorted_indices.transpose(1, 2)  # [batch_size, sequence_length, num_classes]

        # dec_tgtを[batch_size, sequence_length, 1]に拡張
        expanded_dec_tgt = dec_tgt.unsqueeze(2)  # [batch_size, sequence_length, 1]

        # 正解ラベルのランク（位置）を調べる
        ranks = (sorted_indices == expanded_dec_tgt).nonzero(as_tuple=True)[2] + 1  # 1-based rank
        ranks = ranks.reshape(dec_tgt.shape)  # [batch_size, sequence_length]
        print("codon Ranks:", ranks, flush=True)

        # dna rank
        # 各位置での正解ラベル（dec_tgt）がoutput_codonで上から何番目に位置するかを調べる
        sorted_scores, sorted_indices = output_dna.sort(dim=1, descending=True)
        sorted_indices = sorted_indices.transpose(1, 2)  # [batch_size, sequence_length, num_classes]

        # dec_tgtを[batch_size, sequence_length, 1]に拡張
        expanded_dec_tgt = dec_tgt_dna.unsqueeze(2)  # [batch_size, sequence_length, 1]

        # 正解ラベルのランク（位置）を調べる
        ranks = (sorted_indices == expanded_dec_tgt).nonzero(as_tuple=True)[2] + 1  # 1-based rank
        ranks = ranks.reshape(dec_tgt_dna.shape)  # [batch_size, sequence_length]
        print("DNA Ranks:", ranks, flush=True)


    for batch in test_loader:
        dec_tgt = batch[1].to(device)
        dec_tgt_protein = batch[2].to(device) 
        dec_tgt_dna = batch[3].to(device)    
        src = batch[4].to(device)
        src_protein = batch[5].to(device)
        src_dna = batch[6].to(device)

        dec_ipt = torch.tensor([[dataset.vocab['<bos>']]] * len(src), dtype=torch.long, device=device)
        dec_ipt_protein = torch.tensor([[dataset.vocab['<bos>']]] * len(src), dtype=torch.long, device=device)
        dec_ipt_dna = torch.tensor([[dataset.vocab['<bos>']]*3] * len(src), dtype=torch.long, device=device)
        
        for i in range(1000):
            output_codon, output_pro, output_dna, _, _ = model(dec_ipt, dec_ipt_amino, dec_ipt_dna,  src, src_protein, src_dna)
            output_codon = torch.argmax(output_codon, dim=2)
            # 最も確率の高いcodonトークンを取得
            next_item = output_codon[:, -1].unsqueeze(1) 
            # 予測されたcodonトークンをデコーダの入力に追加
            dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

            output_pro =  torch.argmax(output_pro, dim=2)
            # 最も確率の高いトークンを取得
            next_item = output_pro[:, -1].unsqueeze(1) 
            # 予測されたproteinトークンをデコーダの入力に追加
            dec_ipt_protein = torch.cat((dec_ipt_protein, next_item), dim=1)

            output_dna =  torch.argmax(output_dna, dim=2)
            # 最も確率の高いトークンを3塩基分取得
            next_item = output_dna[:, -3:] 
            # 予測されたproteinトークンをデコーダの入力に追加
            dec_ipt_dna = torch.cat((dec_ipt_dna, next_item), dim=1)

            # 文末を表すトークンが出力されたら終了
            # 各行に'<eos>'が含まれているか否かを判定
            end_token_count = (dec_ipt == dataset.vocab['<eos>']).any(dim=1).sum().item()
            # end_token_count = (dec_ipt_dna == dataset.vocab_dna['<eos>']).any(dim=1).sum().item()
            if end_token_count == len(src):
                break
        src = torch.cat((src[:, :1], src[:, 3:]), dim=1)
        source_sequences += alignment.extract_sequences(src)
        target_sequences += alignment.extract_sequences(dec_tgt)
        predicted_sequences += alignment.extract_sequences(dec_ipt)
    
    if args.train:
        if args.horovod:
            source_sequences, target_sequences , predicted_sequences = allreduce(source_sequences, target_sequences, predicted_sequences, dataset.vocab)
            # if hvd.rank() == 0:  # ここでランク0のノードのみが実行
            plot_obj = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
            plot_obj.savefig(os.path.join(result_folder, "align.png"))
            print('\n'.join([''.join([dataset.vocab.index_to_token[codon] for codon in sequence]) for sequence in predicted_sequences]), flush=True)

        else:
            plot_obj = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
            plot_obj.savefig(os.path.join(result_folder, "align.png"))
            print('\n'.join([''.join([dataset.vocab.index_to_token[codon] for codon in sequence]) for sequence in predicted_sequences]), flush=True)

    # print(''.join([dataset.vocab.index_to_token[codon] for sequence in predicted_sequences for codon in sequence ]))
    # print('\n'.join([''.join([dataset.vocab.index_to_token[codon] for codon in sequence]) for sequence in predicted_sequences]), flush=True)

    # scheduler.step()