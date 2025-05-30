import math
import random
import torch
import RNA
import matplotlib.pyplot as plt
import os
import sys
import torch.nn.functional as F
import shutil 

class Node:
    def __init__(self, sequence, dec_ipt, parent=None, depth=0):
        self.sequence = sequence  # 現在のコドン配列
        self.dec_ipt = dec_ipt
        self.parent = parent  # 親ノード
        self.depth = depth
        self.children = []  # 子ノードのリスト
        self.visits = 0  # このノードの訪問回数
        self.value = 0  # このノードの価値

    def is_fully_expanded(self):
        return len(self.children) > 0  # 子ノードが展開されているかチェック

    def best_child(self):
        # UCB1アルゴリズムを使用して最良の子ノードを選択
        return max(self.children, key=lambda node: node.value / (node.visits + 1e-6) +
                                        0.5 * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6)))

    def expand(self, model, vocab, src, len_predicted_seq):
        # ノードが完全に展開されていない場合、新しい子ノードを作成
        if self.visits > 3 and not self.is_fully_expanded() and len(self.sequence) < len_predicted_seq:
            # トークンの確率を取得
            output_codon = model(self.dec_ipt, src)
            # Softmaxを適用して確率を正規化
            probabilities = F.softmax(output_codon, dim=2)

            # 確率に基づいてトークンをソートし、上位3つのインデックスを取得
            top_values, top_indices = torch.topk(probabilities, 3, dim=2)

            for i in range(3):
                # 上位i番目のトークンを取得
                next_item = top_indices[:, -1, i].unsqueeze(1)
                next_value = top_values[:, -1, i].unsqueeze(1)

                # 予測されたトークンとその値を出力
                predicted_token = vocab.index_to_token[next_item[0].item()]
                # print(f"Top {i+1} predicted token: {predicted_token}, value: {next_value[0].item()}")
                
                if next_value[0].item() > 0.1:
                    # 予測されたトークンをデコーダの入力に追加
                    dec_ipt = torch.cat((self.dec_ipt, next_item), dim=1)
                
                    new_sequence = self.sequence + [vocab.index_to_token[next_item[0].item()]]
                    child = Node(new_sequence, dec_ipt, parent=self, depth=self.depth+1)
                    self.children.append(child)

def reward(sequence, target_gc_content=0.365):
    if '<eos>' in sequence:
        sequence.remove('<eos>')
    if '<bos>' in sequence:
        sequence.remove('<bos>')
    if '<pad>' in sequence:
        sequence.remove('<pad>')
    sequence =  ''.join(sequence)
    # 最終的な配列のGCコンテンツに基づいて報酬を計算
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    ss, mfe = RNA.fold(sequence)
    
    gc_content_reward = 20/100**abs(gc_content - target_gc_content)
    mfe_reward = abs(mfe)/20
    # all_reward = gc_content_reward + mfe_reward
    all_reward = gc_content_reward
    return all_reward, mfe, gc_content

def simulate(node, model, src, template_seq, vocab, epoch):
    # ロールアウトフェーズ: ランダムにアクションを選択し、最終状態で報酬を計算
    
    if (epoch % 500) == 0:
        sequence = node.sequence.copy()  # 現在のシーケンスをコピー
        dec_ipt = node.dec_ipt.detach()
        for i in range(1000):
            output_codon = model(dec_ipt, src)
            output_codon = torch.argmax(output_codon, dim=2)
            # 最も確率の高いcodonトークンを取得
            next_item = output_codon[:, -1].unsqueeze(1) 
            # 予測されたcodonトークンをデコーダの入力に追加
            dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

            sequence.append(vocab.index_to_token[next_item[0].item()])

            # 文末を表すトークンが出力されたら終了
            # 各行に'<eos>'が含まれているか否かを判定
            end_token_count = ((dec_ipt == vocab['<eos>']) | (dec_ipt == vocab['<pad>'])).any(dim=1).sum().item()
            if end_token_count == len(src):
                break
    else:
        sequence = node.sequence.copy()  # 現在のシーケンスをコピー
        sequence = sequence + template_seq[len(sequence) : ]

    return sequence  # 最終的なシーケンスに対する報酬を計算

def backpropagate(node, value):
    # バックプロパゲーション: 価値を親ノードに伝播
    while node:
        node.visits += 1  # 訪問回数をインクリメント
        node.value += value  # 価値を更新
        node = node.parent  # 親ノードに移動

def main_loop(root, model, vocab, tgt, src, predicted_seq, result_folder, iterations=1000000):
    rewards = []  # 報酬値を保存するリスト
    gc_contents = []  # GCコンテンツの値を保存するリスト
    mfes = []  # 最低自由エネルギー(MFE)の値を保存するリスト
    tree_depths = []  # ツリーの深さを保存するリスト

    # MCTSのメインループ
    template_seq = predicted_seq.copy()
    for i in range(iterations):
        node = root  # ルートノードからスタート
        while node.is_fully_expanded():
            node = node.best_child()
        node.expand(model, vocab, src, len(predicted_seq))  # ノードの訪問回数が一定のしきい値を超えた場合にのみ展開
        if node.children:
            child = node.children[0]  # 最初に追加された子ノードを取得
        else:
            child = node
        sequence = simulate(child, model, src, template_seq, vocab, i)  # 子ノードに対してシミュレーションを実行

        if (i % 500) == 0:
            template_seq = sequence.copy()

        all_reward, mfe, gc_content = reward(sequence)
        
        backpropagate(child, all_reward)  # 価値をバックプロパゲーション

        rewards.append(all_reward)
        gc_contents.append(gc_content)
        mfes.append(mfe)
        tree_depths.append(child.depth)  # ツリーの深さをリストに追加

        if (i + 1) % 100 == 0:  # 100回ごとにグラフを描画して保存
            print("tree depth =", child.depth)
            print("full seq length =", len(sequence))
            print("full seq =", sequence)
            print("MFE = " , mfe, flush=True)
            print("GC content =", gc_content)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.plot(rewards)
            plt.xlabel('Iterations')
            plt.ylabel('Reward')
            plt.title('Reward over Iterations')

            plt.subplot(1, 4, 2)
            plt.plot(gc_contents)
            plt.xlabel('Iterations')
            plt.ylabel('GC Content')
            plt.title('GC Content over Iterations')

            plt.subplot(1, 4, 3)
            plt.plot(mfes)
            plt.xlabel('Iterations')
            plt.ylabel('MFE')
            plt.title('MFE over Iterations')

            plt.subplot(1, 4, 4)
            plt.plot(tree_depths)
            plt.xlabel('Iterations')
            plt.ylabel('Tree Depth')
            plt.title('Tree Depth over Iterations')

            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, "mcts_metrics.png"))  # PNG形式で保存
            plt.close()


def mcts(model, test_loader, vocab, device, edition_fasta, predicted_sequences, result_folder):
    # shutil.copy("./mcts.py", os.path.join(result_folder, "mcts.py"))
    for batch in test_loader:
        tgt = batch[1].to(device)
        src = batch[2].to(device)
        # src_sequence = [[vocab.index_to_token[codon.item()] for codon in sequence] for sequence in src][0]
        # src_sequence= src_sequence[src_sequence.index('<bos>') + 1 : src_sequence.index('<eos>')]

        dec_ipt = tgt[:, :2]
        sequence=[]
        if edition_fasta:
            dec_ipt = tgt[:,:-1]
            sequence = [[vocab.index_to_token[int(codon)] for codon in sequence] for sequence in dec_ipt][0]
            sequence = sequence[2:]
            print(sequence)
        
        root = Node(sequence=sequence, dec_ipt=dec_ipt)  # 初期状態のノードを作成
        main_loop(root, model, vocab, tgt, src, predicted_sequences[0], result_folder)  # MCTSを実行
        best_child = root.best_child()  # 最良の子ノードを取得
        print(f"Best child sequence: {best_child.sequence}, Value: {best_child.value / best_child.visits}")  # 結果を出力
