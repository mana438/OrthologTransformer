# arg_parser.py
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str)
    # parser.add_argument('--model_output', type=str, default="./model_weights.pth")
    parser.add_argument('--ortholog_files_test', type=str)
    parser.add_argument('--ortholog_files_train', type=str)
    parser.add_argument('--OMA_species', type=str, default="../data/OMA_database/prokaryotes_group.txt")
    parser.add_argument('--result_folder', type=str, default=None, help='Path to result folder')
    parser.add_argument('--pickle_path', type=str)
    parser.add_argument('--edition_fasta', action='store_true')
    

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--calm', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--reverse', type=bool, default=False)

    # hvd
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    # nomal
    # parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--data_alignment',type=bool, default=False)

    parser.add_argument('--use_gap',type=bool, default=False)
    parser.add_argument('--gap_open', type=int, default=-10)
    parser.add_argument('--gap_extend', type=int, default=-1)

    parser.add_argument('--gap_ratio', type=float, default=0.3)
    parser.add_argument('--horovod', type=bool, default=False)
    parser.add_argument('--mcts', type=bool, default=False)

    # model parameter
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=13)
    parser.add_argument('--num_decoder_layers', type=int, default=13)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--memory_mask', type=bool, default=False)
    
    # beam search
    parser.add_argument("--use_beam", action="store_true", help="ビームサーチを使用する場合に指定")
    parser.add_argument("--beam_size", type=int, default=1, help="ビームサーチ時のビーム幅")  # ← 通常の greedy 相当
    parser.add_argument("--num_return_sequences", type=int, default=1, help="ビームサーチで返すシーケンスの数")  # ← 上と合わせる
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Diverse Beam Search のグループ数")  # ← バニラなら 1
    parser.add_argument("--diversity_strength", type=float, default=0.0, help="ビーム間の類似度ペナルティ")  # ← 0.0 = 無効
    parser.add_argument("--temperature", type=float, default=1.0, help="softmax の温度")  # ← 通常ソフトマックス
    parser.add_argument("--sampling_k", type=int, default=0, help="Top-k サンプリング時に選ぶ候補数（0で無効）")
    parser.add_argument("--sampling_steps", type=int, default=0, help="Top-k サンプリングを適用するステップ数（0で無効）")


    args = parser.parse_args()

    return args
