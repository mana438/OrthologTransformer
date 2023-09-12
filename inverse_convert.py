import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchtext.data.metrics import bleu_score
import glob

from util import custom_collate_fn, alignment, count_nonzero_matches
from model import CodonTransformer
from read import Vocab, OrthologDataset

import os
from concurrent.futures import ThreadPoolExecutor

from Bio import pairwise2
from Bio.Seq import Seq

# テスト対象のオルソログ関係ファイルのリスト
ortholog_files_train_test = glob.glob("/home/aca10223gf/workplace/data/sample_ortho/Bacillus_subtilis_168_cds__v__Bacillus_thuringiensis_ATCC_10792_cds.tsv")

# 学習のみのオルソログ関係ファイルのリスト
ortholog_files_train = glob.glob("/home/aca10223gf/workplace/data/sample_ortho/sample_train/*")

# FASTAファイルが格納されているディレクトリ
fasta_dir = "/home/aca10223gf/workplace/data/CDS_fasta"

# OrthologDataset オブジェクトを作成する
dataset = OrthologDataset(fasta_dir)
# テスト対象のオルソログ関係ファイルを読み込む
dataset.load_data(ortholog_files_train_test, False)
print("train_test dataset: " + str(len(dataset)))
# test groupを生成
dataset.split_groups()

########
train_dataset, test_dataset = dataset.split_dataset()
print("train: ", len(train_dataset))
print("test: ", len(test_dataset))
#######

# 学習のみのオルソログ関係ファイルを追加。ただしtest groupは読み込まない
dataset.load_data(ortholog_files_train, True)

print("dataset: " + str(len(dataset)))
# データセットをトレーニングデータとテストデータに分割する
train_dataset, test_dataset = dataset.split_dataset()

#######
print("train: ", len(train_dataset))
print("test: ", len(test_dataset))
#########

# トレーニングデータとテストデータを DataLoader でラップする
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

alignment = alignment(dataset.vocab)

# ハイパーパラメータ
d_model = 512
nhead = 16
num_layers = 10
dim_feedforward = 512

# token size
vocab_size_input = dataset.len_vocab_input()
vocab_size_target =dataset.len_vocab_target()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルインスタンス生成
model = CodonTransformer(vocab_size_input, vocab_size_target, d_model, nhead, num_layers, dim_feedforward).to(device)

# 損失関数と最適化アルゴリズム
# criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
criterion = nn.CrossEntropyLoss()
# criterion =  nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
torch.set_printoptions(edgeitems=10, threshold=10000, linewidth=200)

# 学習
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    # トレーニングのループの外で定義します
    train_correct_predictions = 0
    train_total_predictions = 0

    total_loss = 0
    num_batches = 0

    for batch in train_loader:
        src = batch[2].to(device)
        tgt = batch[1].to(device)        

        dec_ipt = tgt[:, :-1]
        dec_tgt = tgt[:, 1:] #右に1つずらす
        # dec_tgt = nn.functional.one_hot(dec_tgt, vocab_size_target).to(torch.float32).to(device) #MSELossの時
        optimizer.zero_grad()
        output = model(src, dec_ipt)
        output = output.transpose(1, 2) #CrossEntropyLossの時
        loss = criterion(output, dec_tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
  
        if (epoch + 1) % 1 == 0:
            # 正解数と全体の予測数を計算
            output = output.transpose(1, 2)
            output = torch.argmax(output, dim=2)
            match_count, all_count = count_nonzero_matches(tgt[:, 1:], output)
            train_correct_predictions += match_count
            train_total_predictions += all_count

            # for tgt_seq, pred_seq in zip(tgt[:, 1:], output):
            #     print("print seq")
            #     print("tgt", tgt_seq)
            #     print("pred", pred_seq)
 
    # 各エポックの最後に、トレーニングの分類予測精度を表示します。
    train_accuracy = train_correct_predictions / train_total_predictions
    print(f"Training Accuracy: {train_accuracy:.4f}")
        
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")



    # 評価
    if (epoch + 1) % 100 == 0:
        model.eval()
        total_alignment_score = 0.0

        source_sequences = []
        target_sequences = []
        predicted_sequences = []
        
        with torch.no_grad():

            for batch in test_loader:
                src = batch[2].to(device)
                dec_tgt = batch[1].to(device)
                dec_ipt = torch.tensor([[dataset.vocab['<bos>']]] * len(src), dtype=torch.long, device=device)

                
                for i in range(1000):
                    output = model(src, dec_ipt)
                    output = torch.argmax(output, dim=2)
                    # 最も確率の高いトークンを取得
                    next_item = output[:, -1].unsqueeze(1) 

                    # 予測されたトークンをデコーダの入力に追加
                    dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

                    # 文末を表すトークンが出力されたら終了
                    # 各行に'<eos>'が含まれているか否かを判定
                    end_token_count = (dec_ipt == dataset.vocab['<eos>']).any(dim=1).sum().item()
                    if end_token_count == len(src):
                        break
                src = torch.cat((src[:, :1], src[:, 3:]), dim=1)
                source_sequences += alignment.extract_sequences(src)
                target_sequences += alignment.extract_sequences(dec_tgt)
                predicted_sequences += alignment.extract_sequences(dec_ipt)
       
        alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)

    # scheduler.step()

filename = "/home/aca10223gf/workplace/job_results/weight/model_weights.pth"
counter = 1
while os.path.exists(filename):
    # ファイル名の拡張子の前にカウンタを追加する
    filename = f"/home/aca10223gf/workplace/job_results/weight/model_weights_{counter}.png"
    counter += 1
    # モデルの state_dict を保存
    torch.save(model.state_dict(), filename)