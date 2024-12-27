import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
#from torchtext.data.metrics import bleu_score
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, count_parameters, CG_ratio
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor

from Bio.Seq import Seq

import traceback
import random

import json
import argparse
import glob
from arg_parser import get_args
import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from calm.sequence import CodonSequence  # CodonSequenceクラス
from calm.alphabet import Alphabet  # アルファベット定義
from calm.model import ProteinBertModel
from calm.pretrained import ARGS  # 事前に定義されたARGSを使用

args = get_args()
if args.horovod:
    import horovod.torch as hvd
    hvd.init()
else:
    hvd = None

# GradScalerの初期化
scaler = GradScaler()

# 結果出力フォルダー
import datetime
result_folder = os.path.join("../job_results/", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(result_folder, exist_ok=True)

# パラメータ出力
with open(os.path.join(result_folder, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

# 標準出力先の変更
# sys.stdout = open(os.path.join(result_folder, "output.txt"), "w")

# OMA_speciesファイル
OMA_species = args.OMA_species
# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(OMA_species,"./vocab_OMA.json")
# テスト対象のオルソログ関係ファイルのリスト
ortholog_files_test = glob.glob(args.ortholog_files_test.replace("'", ""))
test_dataset = dataset.load_data(ortholog_files_test, False, args.calm)

if args.train:
    # 学習のみのオルソログ関係ファイルのリスト
    ortholog_files_train = glob.glob(args.ortholog_files_train.replace("'", ""))
    

    #ortholog_files_train = ortholog_files_train * 100

    print(len(ortholog_files_train))
    start = time.time()  # 時間計測開始
#    train_dataset = dataset.load_data(ortholog_files_train, args.reverse, args.calm)
    print(f"Data loading took 1{time.time() - start:.2f} seconds.", flush=True)

    # ファイルを指定した数で順序を保ったまま分割
    def split_files(files, n_groups):
        chunk_size = (len(files) + n_groups - 1) // n_groups  # 切り上げで計算
        return [files[i * chunk_size:(i + 1) * chunk_size] for i in range(n_groups)]

    start = time.time()
    max_workers = os.cpu_count()
    file_chunks = split_files(ortholog_files_train, max_workers)
    
    # 並列処理で各チャンクをロード
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        train_dataset = sum(executor.map(lambda chunk: dataset.load_data(chunk, args.reverse, args.calm), file_chunks), [])

    print(f"Data loading took {time.time() - start:.2f} seconds.", flush=True)

#######

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
    # print(f"Process {hvd.rank()}: Using GPU {hvd.local_rank()} - {torch.cuda.get_device_name(hvd.local_rank())} ({torch.cuda.get_device_properties(hvd.local_rank()).total_memory / 1e9:.2f} GB)", flush=True)
    print(f"Process {hvd.rank()}: Local rank {hvd.local_rank()}", flush=True)

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
    # calm_modelのパラメータをフリーズ
    for param in calm_model.parameters():
        param.requires_grad = False
    if args.horovod:
        hvd.broadcast_parameters(calm_model.state_dict(), root_rank=0)


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
        start_time = time.time()  # エポック開始時の時間を記録
        try:
            print(epoch, " epoch start", flush=True)
            model.train()
            # トレーニングのループの外で定義します
            train_correct_predictions = 0
            train_total_predictions = 0

            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                tgt = batch[1].to(device)                
                src = batch[2].to(device)

                dec_ipt = tgt[:, :-1]
                dec_tgt = tgt[:, 1:] #右に1つずらす
                
                with autocast():  # FP16演算を有効化
                    if args.calm:
                        calm_output = calm_model(batch[3].to(device), repr_layers=[12])["representations"][12].detach()
                    else:
                        calm_output = None
                    output_codon = model(dec_ipt, src, calm_output)
                    output_codon = output_codon.transpose(1, 2) #CrossEntropyLossの時
                    loss_codon = criterion(output_codon, dec_tgt)
                    loss = loss_codon
                
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

# 評価
model.eval()
total_alignment_score = 0.0

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

        for i in range(700):
            with autocast():  # 評価でもFP16演算を有効化
                output_codon_logits = model(dec_ipt, src, calm_model(batch[3].to(device), repr_layers=[12])["representations"][12] if args.calm else None)
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
    
    #if args.train:
    if args.horovod:            
        source_sequences, target_sequences , predicted_sequences = allreduce(source_sequences, target_sequences, predicted_sequences, dataset.vocab)
        # if hvd.rank() == 0:  # ここでランク0のノードのみが実行
    plot_obj, score_ratio = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
    plot_obj.savefig(os.path.join(result_folder, "align.png"))

    predicted_sequences = [[dataset.vocab.index_to_token[codon] for codon in sequence] for sequence in predicted_sequences]
    predicted_sequences_all = [''.join(sequence) for sequence in predicted_sequences]
    print("GC ratio " +  str(CG_ratio(predicted_sequences_all)))
    print('\n'.join(predicted_sequences_all))
    print(score_ratio)

    if args.mcts:
        mcts(model, test_loader, dataset.vocab, device, args.edition_fasta, predicted_sequences, args.memory_mask, result_folder)
