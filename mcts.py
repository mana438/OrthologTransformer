import math
import random
from your_transformer_model import TransformerModel  # Transformerモデルをインポートします

class Node:
    def __init__(self, sequence, parent=None):
        self.sequence = sequence  # 現在のコドン配列
        self.parent = parent  # 親ノード
        self.children = []  # 子ノードのリスト
        self.visits = 0  # このノードの訪問回数
        self.value = 0  # このノードの価値

    def is_fully_expanded(self):
        return len(self.children) == 3  # 上位3候補の子ノードが展開されているかチェック

    def best_child(self):
        # UCB1アルゴリズムを使用して最良の子ノードを選択
        return max(self.children, key=lambda node: node.value / (node.visits + 1e-6) +
                                       math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6)))

    def expand(self, model):
        # ノードが完全に展開されていない場合、新しい子ノードを作成
        if self.visits > SOME_THRESHOLD and not self.is_fully_expanded():
            top_3_codons = model.predict_top_3(self.sequence)
            for codon in top_3_codons:
                new_sequence = self.sequence + [codon]
                child = Node(new_sequence, parent=self)
                self.children.append(child)

def reward(sequence, target_gc_content=0.5):
    # 最終的な配列のGCコンテンツに基づいて報酬を計算
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    return -abs(gc_content - target_gc_content)  # 目標GCコンテンツからの差を報酬とする

def simulate(node, model):
    # ロールアウトフェーズ: ランダムにアクションを選択し、最終状態で報酬を計算
    sequence = node.sequence.copy()  # 現在のシーケンスをコピー
    for _ in range(10):  # 最大10回のアクションをシミュレート
        top_3_codons = model.predict_top_3(sequence)  # 上位3候補のコドンを予測
        chosen_codon = random.choice(top_3_codons)  # ランダムにコドンを選択
        sequence.append(chosen_codon)  # 選択したコドンをシーケンスに追加
    return reward(sequence)  # 最終的なシーケンスに対する報酬を計算

def backpropagate(node, value):
    # バックプロパゲーション: 価値を親ノードに伝播
    while node:
        node.visits += 1  # 訪問回数をインクリメント
        node.value += value  # 価値を更新
        node = node.parent  # 親ノードに移動

def mcts(root, model, iterations=1000):
    # MCTSのメインループ
    for _ in range(iterations):
        node = root  # ルートノードからスタート
        while node.is_fully_expanded():
            node = node.best_child()
        node.expand(model)  # ノードの訪問回数が一定のしきい値を超えた場合にのみ展開
        child = node.children[-1]  # 最後に追加された子ノードを取得
        value = simulate(child, model)  # 子ノードに対してシミュレーションを実行
        backpropagate(child, value)  # 価値をバックプロパゲーション

if __name__ == "__main__":
    model = TransformerModel()  # Transformerモデルのインスタンスを作成
    root = Node(sequence=[])  # 初期状態のノードを作成
    mcts(root, model)  # MCTSを実行
    best_child = root.best_child()  # 最良の子ノードを取得
    print(f"Best child sequence: {best_child.sequence}, Value: {best_child.value / best_child.visits}")  # 結果を出力
