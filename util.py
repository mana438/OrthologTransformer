from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import os
import random
import statistics
import numpy as np
import json
import copy
from functools import partial


class alignment:
    def  __init__(self, vocab):
        self.vocab = vocab
        # Dictionary mapping numbers 3 to 66 to arbitrary single characters
        self.num_to_char = {}
        self.char_to_num = {}
        start_ascii = 33  # ASCII code for '!'
        for i in range(3, 67):
            char = chr(start_ascii + i - 3)
            if char == '-':
                start_ascii += 1
                char = chr(start_ascii + i - 3)
            self.num_to_char[i] = char
            self.char_to_num[char] = i

    # Extract sequences between 1 and 2  
    def extract_sequences(self, tensor):
        result = []
        for row in tensor:
            # start_idx = (row == 1).nonzero(as_tuple=True)[0][0] + 1
            one_indices = (row == 1).nonzero(as_tuple=True)[0]
            # If 1 does not exist, start extraction from the beginning of the row
            start_idx = one_indices[0] + 1 if one_indices.size(0) > 0 else 0
        
            end_indices_2 = (row == 2).nonzero(as_tuple=True)[0]
            end_indices_0 = (row == 0).nonzero(as_tuple=True)[0]

            if end_indices_2.size(0) == 0:
                end_idx_2 = row.size(0)
            else:
                end_idx_2 = end_indices_2[0]

            if end_indices_0.size(0) == 0:
                end_idx_0 = row.size(0)
            else:
                end_idx_0 = end_indices_0[0]

            end_idx = min(end_idx_2, end_idx_0)

            # Remove 0, 1, and 2 from the extracted sequence
            extracted_sequence = [num for num in row[start_idx:end_idx].tolist() if num not in [0, 1, 2]]
            result.append(extracted_sequence)
        return result


    # Function to convert a list of numbers into a string
    def convert_num_list_to_string(self, num_list, num_to_char):
        return ''.join([self.num_to_char[num] for num in num_list])


    # Function to align sequences
    def align_sequences(self, seq_pair):
        seq1, seq2 = seq_pair
        seq1 = ''.join([self.num_to_char[num] for num in seq1])
        seq2 = ''.join([self.num_to_char[num] for num in seq2])  
          
        alignments = pairwise2.align.globalxx(Seq(seq1), Seq(seq2))

        if not alignments:
            print(f"Warning: No alignments found for sequences: {seq1}, {seq2}")
            return seq1, seq2, 0

        aligned_seq1, aligned_seq2, score, _, _ = alignments[0]
        aligned_seq1 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq1) ])
        aligned_seq2 = ''.join([self.vocab.index_to_token[self.char_to_num[char]] if char != '-' else "---" for char in Seq(aligned_seq2)])
        

        return Seq(aligned_seq1), Seq(aligned_seq2), score

    # align structure
    def align_structures(self, seq_pair):
        seq1, seq2 = seq_pair
        seq1 = ''.join([self.vocab.index_to_token[num] for num in seq1])
        seq2 = ''.join([self.vocab.index_to_token[num] for num in seq2])  
        
    def align_sequences_aa(self, pair):
        s1, s2 = pair
        # BLOSUM62
        blosum62 = substitution_matrices.load("BLOSUM62")
        
        gap_open = -10
        gap_extend = -0.5
    
        # globalds: global alignment with a scoring dictionary
        alignments = pairwise2.align.globalds(s1, s2, blosum62, gap_open, gap_extend)
    
        # Get one best alignment (multiple candidates may be returned)
        best_alignment = alignments[0]
        aligned_s1 = best_alignment.seqA
        aligned_s2 = best_alignment.seqB
        score = best_alignment.score
        return aligned_s1, aligned_s2, score
        
    def predict_structure(self, dna_sequence):
        dna_seq = Seq(dna_sequence)
        amino_acid_seq = dna_seq.translate()

        # Save amino acid sequence in FASTA format
        fasta_file_path = '../job_result/query.fasta'
        record = SeqRecord(amino_acid_seq, id="query_sequence", description="Translated sequence")
        os.makedirs(os.path.dirname(fasta_file_path), exist_ok=True)  # ディレクトリが存在しない場合に作成
        with open(fasta_file_path, 'w') as fasta_file:
            SeqIO.write(record, fasta_file, "fasta")

        # Path to execution script
        alphafold_script_path = '../alphafold/run_alphafold.sh'
        # Run shell command without changing directory
        result = subprocess.run([
            alphafold_script_path,
            '-a', '0,1,2,3',
            '-d', os.getenv('ALPHAFOLD_DATA_DIR'),
            '-o', '../dummy_test/',
            '-m', 'model_1',
            '-f', fasta_file_path,
            '-t', '2020-05-14'
        ], capture_output=True, text=True)

        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)




    def chunks(self, lst, chunk_size):
        """Generator to split a list into chunks of specified size"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def plot_alignment_scores(self, src_seqs, tgt_seqs, pred_seqs, xlabel="Alignment score of source and target codons", ylabel="Alignment score of target and predicted codons"):
        src_tgt_scores = []
        tgt_pred_scores = []
        score_ratio = []
        # Take 10 elements from each list to form tuples and create a list of those tuples
        batch_num = 30
        src_seqs = list(self.chunks(src_seqs, batch_num))
        tgt_seqs = list(self.chunks(tgt_seqs, batch_num))
        pred_seqs = list(self.chunks(pred_seqs, batch_num))

        for src_seq, tgt_seq, pred_seq in zip(src_seqs, tgt_seqs, pred_seqs):
            # Pairwise alignment of source_sequences and target_sequences
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_src_tgt = list(executor.map(self.align_sequences, zip(src_seq, tgt_seq)))
            aligned_src, aligned_tgt, src_tgt_score = zip(*results_src_tgt)

            # Pairwise alignment of target_sequences and predicted_sequences
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_tgt_pred = list(executor.map(self.align_sequences, zip(tgt_seq, pred_seq)))
            aligned_tgt, aligned_pred, tgt_pred_score = zip(*results_tgt_pred)
            # self.print_alignment(aligned_tgt, aligned_pred, tgt_pred_score)


            # Normalize alignment scores by sequence length (here, length of tgt_seq)
            src_tgt_score_corrected = [score / len(tgt) if len(tgt) > 0 else 0 
                                       for score, tgt in zip(src_tgt_score, tgt_seq)]
            tgt_pred_score_corrected = [score / len(tgt) if len(tgt) > 0 else 0 
                                        for score, tgt in zip(tgt_pred_score, tgt_seq)]
            score_ratio += [pred_score / src_score if src_score != 0 else 0 for pred_score, src_score in zip(tgt_pred_score_corrected, src_tgt_score_corrected)]

            src_tgt_scores += src_tgt_score_corrected
            tgt_pred_scores += tgt_pred_score_corrected
        
        plt.figure()
        # Plot scatter plot
        plt.scatter(src_tgt_scores, tgt_pred_scores)

        src_tgt_mean = np.mean(src_tgt_scores)
        tgt_pred_mean = np.mean(tgt_pred_scores)

        ax = plt.gca()  # Get current Axes object

        # Display average of src_tgt_scores at bottom-right
        plt.text(0.95, 0.05, f'src-tgt Ave: {src_tgt_mean:.3f}', 
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=12)

        # Display average of tgt_pred_scores at top-left
        plt.text(0.05, 0.95, f'tgt-pred Ave: {tgt_pred_mean:.3f}', 
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, fontsize=12)
        
        # Calculate min and max values for x and y axes
        min_value = min(min(src_tgt_scores), min(tgt_pred_scores))
        max_value = max(max(src_tgt_scores), max(tgt_pred_scores))

        # min_value = 0.2
        # max_value = 0.6

        
        # Plot x=y line
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt, score_ratio, src_tgt_mean, tgt_pred_mean
        
        
    def plot_alignment_scores_aa(self, src_seqs, tgt_seqs, pred_seqs, 
                                xlabel="Alignment score of source and target AAs", 
                                ylabel="Alignment score of target and predicted AAs"):
        src_tgt_scores = []
        tgt_pred_scores = []
        score_ratio = []

        # Chunk size for list splitting
        batch_num = 30
        src_seqs = list(self.chunks(src_seqs, batch_num))
        tgt_seqs = list(self.chunks(tgt_seqs, batch_num))
        pred_seqs = list(self.chunks(pred_seqs, batch_num))

        for src_seq, tgt_seq, pred_seq in zip(src_seqs, tgt_seqs, pred_seqs):
            # Perform pairwise alignment of source_sequences and target_sequences
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_src_tgt = list(executor.map(self.align_sequences_aa, zip(src_seq, tgt_seq)))
            aligned_src, aligned_tgt, src_tgt_score = zip(*results_src_tgt)

            # Perform pairwise alignment of target_sequences and predicted_sequences
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results_tgt_pred = list(executor.map(self.align_sequences_aa, zip(tgt_seq, pred_seq)))
            aligned_tgt, aligned_pred, tgt_pred_score = zip(*results_tgt_pred)

            # Normalize alignment scores by sequence length (here, length of tgt_seq)
            src_tgt_score_corrected = [score / len(tgt) if len(tgt) > 0 else 0 
                                       for score, tgt in zip(src_tgt_score, tgt_seq)]
            tgt_pred_score_corrected = [score / len(tgt) if len(tgt) > 0 else 0 
                                        for score, tgt in zip(tgt_pred_score, tgt_seq)]

            score_ratio += [pred_score / src_score if src_score != 0 else 0 
                            for pred_score, src_score in zip(tgt_pred_score_corrected, src_tgt_score_corrected)]

            src_tgt_scores += src_tgt_score_corrected
            tgt_pred_scores += tgt_pred_score_corrected
        
        plt.figure()
        # Plot scatter plot
        plt.scatter(src_tgt_scores, tgt_pred_scores)

        src_tgt_mean = np.mean(src_tgt_scores)
        tgt_pred_mean = np.mean(tgt_pred_scores)

        ax = plt.gca()
        plt.text(0.95, 0.05, f'src-tgt Ave: {src_tgt_mean:.3f}', 
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax.transAxes, fontsize=12)
        plt.text(0.05, 0.95, f'tgt-pred Ave: {tgt_pred_mean:.3f}', 
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes, fontsize=12)
        
        min_value = min(min(src_tgt_scores), min(tgt_pred_scores))
        max_value = max(max(src_tgt_scores), max(tgt_pred_scores))
        
        plt.plot([min_value, max_value], [min_value, max_value], 'r--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        return plt, score_ratio, src_tgt_mean, tgt_pred_mean

    
    def print_alignment(self, aligned_seq1, aligned_seq2, score):
        # Display aligned sequences
        for aligned_seq1, aligned_seq2, score in zip(aligned_sequences1, aligned_sequences2, scores):
            print("score: " + str(score))
            print("target sequences")
            print(aligned_seq1)
            print("predicted sequences")
            print(aligned_seq2)
            print("-------")
            total_alignment_score += score
        num_samples += len(src)


# def custom_collate_fn(batch):
#     ortholog_groups, targets, inputs = [], [], []
#     tgt_ids, src_ids = [], [] 

#     for item in batch:
#         # Decompose elements of the batch
#         ortholog_group, tgt, inp, tgt_id, src_id = item[:5]  # 最初の3つを確実に取得
#         ortholog_groups.append(ortholog_group)
#         targets.append(torch.tensor(tgt))
#         inputs.append(torch.tensor(inp))
#         tgt_ids.append(tgt_id)
#         src_ids.append(src_id)
#     # Apply padding
#     targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
#     inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
#     return ortholog_groups, targets_padded, inputs_padded, tgt_ids, src_ids



def custom_collate_fn(batch, mode="train", max_len=700):
    ortholog_groups, targets, inputs = [], [], []
    tgt_ids, src_ids = [], []

    for item in batch:
        ortholog_group, tgt, inp, tgt_id, src_id = item[:5]
        ortholog_groups.append(ortholog_group)
        # トークンIDなら long を明示（必要に応じて調整）
        targets.append(torch.tensor(tgt, dtype=torch.long))
        inputs.append(torch.tensor(inp, dtype=torch.long))
        tgt_ids.append(tgt_id)
        src_ids.append(src_id)

    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    inputs_padded  = pad_sequence(inputs,  batch_first=True, padding_value=0)

    if mode == "test":
        # すでに pad_sequence 済み（バッチ内の最大長）→ max_len まで追加パディング
        # 「必ず max_len 未満」の前提なので切り捨て分岐は不要
        t_pad_len = max_len - targets_padded.size(1)
        i_pad_len = max_len - inputs_padded.size(1)

        # 念のためアサート（開発中の安全網）
        assert t_pad_len > 0 and i_pad_len > 0, "系列長が max_len 以上になっています。"

        targets_padded = F.pad(targets_padded, (0, t_pad_len), value=0)
        inputs_padded  = F.pad(inputs_padded,  (0, i_pad_len), value=0)

    return ortholog_groups, targets_padded, inputs_padded, tgt_ids, src_ids


def count_nonzero_matches(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "The tensors are not the same shape"

    # Create a boolean tensor where matching elements are True, others False
    matches = tensor1 == tensor2

    # Ignore matches of 0 by targeting only non-zero elements
    nonzero_matches = matches & (tensor1 != 0) & (tensor2 != 0)

    # Count the number of matches among non-zero elements
    nonzero_match_count = nonzero_matches.sum().item()

    # Count the number of non-zero element pairs in tensor1 and tensor2
    nonzero_element_pairs = (tensor1 != 0)
    nonzero_element_pair_count = nonzero_element_pairs.sum().item()

    return nonzero_match_count, nonzero_element_pair_count
    
# Load parameters
def load_params(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params


def check_condition(args, hvd):
    if args.horovod:
        if hvd.rank() == 0:
            return True
        else:
            return False
    else:
        return True

def allreduce(source_sequences, target_sequences, predicted_sequences, vocab):
    import horovod.torch as hvd
    # Pad sequences to match the maximum length
    source_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in source_sequences]
    target_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in target_sequences]
    predicted_sequences = [seq + [vocab['<pad>']] * (1000 - len(seq)) for seq in predicted_sequences]

    # Existing hvd.allgather code
    all_source_sequences = hvd.allgather(torch.tensor(source_sequences))
    all_target_sequences = hvd.allgather(torch.tensor(target_sequences))
    all_predicted_sequences = hvd.allgather(torch.tensor(predicted_sequences))

    # Remove padding
    all_source_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_source_sequences]
    all_target_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_target_sequences]
    all_predicted_sequences = [seq[:seq.tolist().index(vocab['<pad>'])].tolist() if vocab['<pad>'] in seq else seq.tolist() for seq in all_predicted_sequences]
    return  all_source_sequences, all_target_sequences , all_predicted_sequences

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def CG_ratio(sequences):
    gc_contents = [ (seq.count('G') + seq.count('C')) / len(seq) for seq in sequences]
    return sum(gc_contents) / len(gc_contents)


def save_with_unique_filename(result_folder, base_name, plot_obj):
    """
    Generate a unique file path and save the plot
    :param result_folder: Folder to save to
    :param base_name: Base file name (e.g., "align.png")
    :param plot_obj: Plot object to save
    """
    output_path = os.path.join(result_folder, base_name)
    for counter in range(1, 1000):
        if not os.path.exists(output_path):
            break
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(result_folder, f"{name}_{counter}{ext}")
    plot_obj.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    
def beam_search(model, src, start_tokens, pad_idx, eos_idx, beam_size, max_len, src_key_padding_mask, num_return_sequences, num_beam_groups, diversity_strength, temperature, sampling_k, sampling_steps):

    assert beam_size % num_beam_groups == 0, "beam_size must be divisible by num_beam_groups"
    assert num_return_sequences <= beam_size, "num_return_sequences must be <= beam_size"

    N = src.size(0)
    device = src.device
    beams = [(start_tokens, torch.zeros(N, device=device))]  # Initial beam
    finished = [[] for _ in range(N)]  # Store completed beams for each sample

    group_size = beam_size // num_beam_groups
    
    step = 0
    for _ in range(max_len):
        all_candidates = []

        for group_id in range(num_beam_groups):
            group_offset = group_id * group_size
            group_tokens = beams[group_offset:group_offset + group_size]

            for tokens, logp in group_tokens:
                expanding = ~(tokens[:, -1] == eos_idx)
                if not expanding.any():
                    all_candidates.append((tokens, logp))
                    continue

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(tokens, src, src_key_padding_mask=src_key_padding_mask)

                log_probs = F.log_softmax(logits[:, -1] / temperature, dim=-1)


                # Add diversity penalty
                if group_id > 0:
                    prev_tokens = [cand[0][:, -1] for cand in all_candidates[-group_size:]]
                    if prev_tokens:
                        prev_tokens = torch.cat(prev_tokens, dim=0)  # (group_size * N,)
                        penalty = torch.zeros_like(log_probs)
                        for t in prev_tokens.unique():
                            penalty[:, t] -= diversity_strength
                        log_probs += penalty

                if step < sampling_steps:
                    for _ in range(group_size):
                        # Perform Top-k sampling separately for each beam
                        topk_probs, topk_idx = torch.topk(log_probs.exp(), sampling_k, dim=-1)  # (N, k)
                        sampled_token_idx = torch.multinomial(topk_probs, num_samples=1).squeeze(1)  # (N,)
                        next_tok = topk_idx[torch.arange(N), sampled_token_idx].unsqueeze(1)  # (N,1)
                        new_seq = torch.cat([tokens, next_tok], dim=1)
                        new_logp = logp + torch.log(topk_probs[torch.arange(N), sampled_token_idx])
                        all_candidates.append((new_seq, new_logp))
                        

                else:
                    topk_logp, topk_idx = torch.topk(log_probs, group_size, dim=-1)
                    for b in range(group_size):
                        next_tok = topk_idx[:, b].unsqueeze(1)
                        new_seq = torch.cat([tokens, next_tok], dim=1)
                        new_logp = logp + topk_logp[:, b]
                        all_candidates.append((new_seq, new_logp))


        # Select top beam_size sequences (simple evaluation by average score)
        scores = torch.stack([cand[1].mean() for cand in all_candidates])
        # top_scores, idx = torch.topk(scores, beam_size)
        top_scores, idx = torch.topk(scores, min(len(scores), beam_size))
        beams = [all_candidates[i] for i in idx]

        # Add beams ending with EOS to finished
        for seq, lp in beams:
            ended = (seq[:, -1] == eos_idx)
            for i in range(N):
                if ended[i]:
                    finished[i].append((seq[i], lp[i].item()))

        if all(finished[i] for i in range(N)):
            break
        
        step += 1
        
    # Return num_return_sequences for each sample
    outputs = []
    for i in range(N):
        seqs = [s for s, _ in sorted(finished[i], key=lambda x: x[1], reverse=True)]
        if len(seqs) < num_return_sequences:
            for seq, lp in beams:
                seqs.append(seq[i])
                if len(seqs) == num_return_sequences:
                    break
        outputs.append(seqs[:num_return_sequences])
    return outputs  # List[N] of List[Tensor]
