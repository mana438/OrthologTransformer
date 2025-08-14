# arg_parser.py
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str)
    parser.add_argument('--ortholog_files_test', type=str)
    parser.add_argument('--ortholog_files_train', type=str)
    
    parser.add_argument('--result_folder', type=str, default=None, help='Path to result folder')
    parser.add_argument('--pickle_path', type=str)
    parser.add_argument('--edition_fasta', action='store_true')
    

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--reverse', type=bool, default=False)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--horovod', type=bool, default=False)
    parser.add_argument('--mcts', type=bool, default=False)

    # model parameter
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--num_encoder_layers', type=int, default=20)
    parser.add_argument('--num_decoder_layers', type=int, default=20)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--memory_mask', type=bool, default=False)
    
    # beam search
    parser.add_argument("--use_beam", action="store_true", help="Specify when using beam search")
    parser.add_argument("--beam_size", type=int, default=20, help="Beam width for beam search")  # ← Equivalent to normal greedy
    parser.add_argument("--num_return_sequences", type=int, default=20, help="Number of sequences returned in beam search")  # ← Match with above
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of groups for Diverse Beam Search")  # ← 1 for vanilla
    parser.add_argument("--diversity_strength", type=float, default=0.0, help="Similarity penalty between beams")  # ← 0.0 = disabled
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")  # ← Normal softmax
    parser.add_argument("--sampling_k", type=int, default=0, help="Number of candidates to select in Top-k sampling (0 to disable)")
    parser.add_argument("--sampling_steps", type=int, default=0, help="Number of steps to apply Top-k sampling (0 to disable)")
    
    args = parser.parse_args()
    
    # If use_beam is False, set related values to None
    if not args.use_beam:
        args.beam_size = None
        args.num_return_sequences = None
        args.num_beam_groups = None
        args.diversity_strength = None
        args.temperature = None
        args.sampling_k = None
        args.sampling_steps = None

    return args
