# arg_parser.py
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input', type=str)
    # parser.add_argument('--model_output', type=str, default="/home/aca10223gf/workplace/job_results/weight/model_weights.pth")
    parser.add_argument('--ortholog_files_test', type=str)
    parser.add_argument('--ortholog_files_train', type=str)
    parser.add_argument('--OMA_species', type=str, default="/home/aca10223gf/workplace/data/OMA_database/prokaryotes_group.txt")
    parser.add_argument('--fasta_dir', type=str, default="/home/aca10223gf/workplace/data/CDS_dna")
    parser.add_argument('--edition_fasta', type=str)
    

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--test_ratio', type=float, default=0.1)
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

    args = parser.parse_args()

    return args
