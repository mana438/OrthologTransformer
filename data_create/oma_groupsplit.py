import networkx as nx
import random
import re
from networkx.algorithms.flow import shortest_augmenting_path

# グラフを作成する。
G = nx.Graph()

# 全オルソロググループの数。仮に1,251,567とします。
total_groups = 1251567

# ファイルから関係を読み込む。
with open('/home/aca10223gf/workplace/data/OMA_database/oma-close-groups.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(':')
        group = parts[0].strip()
        if len(parts) > 1:
            # related_groups = parts[1].strip().split(',')
            # "OMA"で始まる文字列を見つけるための正規表現パターン
            pattern = r'OMAGrp\d+'

            # 正規表現を使用してパターンに一致するすべての文字列をリスト化
            matches = re.findall(pattern, parts[1])
            for match in matches:
                G.add_edge(group, match, capacity=1)
            


# 連結成分を取得し、大きさに基づいて降順でソート
components = sorted(nx.connected_components(G), key=len, reverse=True)

for i, component in enumerate(components[:40]):  # 上位40個の連結成分に対して実行
    print(f"Component {i+1}: {len(component)} nodes")

# すでにグラフに追加されているグループの集合を作成する。
included_groups = set(G.nodes)



# 9:1の割合でグループを分割するための大きいグループと小さいグループのリストを作成する。
large_group = []
small_group = []


# 連結成分ごとに大きいグループまたは小さいグループに割り当てる。
for i, component in enumerate(components):
    if random.random() < 0.7:
        large_group.extend(component)
    else:
        small_group.extend(component)

# ファイルに記載のない単独のオルソロググループをランダムに割り当てる。
for group_number in range(1, total_groups + 1):
    group_id = f"OMAGrp{group_number:06d}"
    if group_id not in included_groups:
        if random.random() < 0.7:
            large_group.append(group_id)
        else:
            small_group.append(group_id)

# 結果をファイルに出力する。
with open('/home/aca10223gf/workplace/data/OMA_database/group_division.txt', 'w') as output_file:
    output_file.write("Large Group:\n")
    output_file.write("\n".join(large_group))
    output_file.write("\n\nSmall Group:\n")
    output_file.write("\n".join(small_group))
