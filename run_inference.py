import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.amp import autocast

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, count_parameters, CG_ratio, save_with_unique_filename, beam_search
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import time
import sys
from pathlib import Path
from datetime import datetime

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import traceback
import random

import json
import argparse
import glob
from arg_parser import get_args
import pickle

from functools import partial


args = get_args()

# Decide and create result folder (default: ./results/<YYYYmmdd_HHMMSS>)
if args.result_folder:
    result_folder = args.result_folder
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_folder = os.path.join(
        script_dir, "results", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
Path(result_folder).mkdir(parents=True, exist_ok=True)

with open(os.path.join(result_folder, "args_test.json"), "w") as f:
    json.dump(vars(args), f, indent=4)


script_dir = os.path.dirname(os.path.abspath(__file__))
default_species_path = os.path.join(script_dir, "prokaryotes_group.txt")
default_vocab_path = os.path.join(script_dir, "vocab_OMA.json")

# Create an OrthologDataset object
dataset = OrthologDataset(default_species_path,default_vocab_path)
# List of ortholog relationship files for testing
ortholog_files_test = sorted(glob.glob(args.ortholog_files_test.replace("'", "")))
# ortholog_files_test = glob.glob(args.ortholog_files_test.replace("'", ""))

test_dataset = dataset.load_data(ortholog_files_test, False)
print("test: ", len(test_dataset), flush=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=partial(custom_collate_fn, mode="test", max_len=700))

# Map to 'cuda:0' if GPU is available, otherwise map to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
# token size
vocab_size_target = dataset.len_vocab_target()
vocab_size_input = dataset.len_vocab_input()

# Create a model instance
model = CodonTransformer(vocab_size_target, vocab_size_input, args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout).to(device)
# model = torch.compile(model, backend="aot_eager")

# Load weights
if args.model_input:
    weights = torch.load(args.model_input, map_location=device)
    model.load_state_dict(weights, strict=True)

print(f'Total number of parameters: {count_parameters(model)}', flush=True)


# model = torch.compile(model, backend="aot_eager")
# model = torch.compile(model)

# evaluation
model.eval()
total_alignment_score = 0.0
alignment = alignment(dataset.vocab)
pad_idx = dataset.vocab['<pad>']

source_sequences = []
target_sequences = []
predicted_sequences = []
source_ids, target_ids = [], [] 

# 1) Obtain EOS token index before starting the loop
eos_idx = dataset.vocab['<eos>']

with torch.no_grad():
    for batch in test_loader:
        tgt = batch[1].to(device)
        src = batch[2].to(device)
        tgt_ids_batch = batch[3]
        src_ids_batch = batch[4] 

        dec_ipt = tgt[:, :2]
        if args.edition_fasta:
            dec_ipt = tgt[:, :-1]

        # src_key_padding_mask = (src == pad_idx).to(device)

        # --- Beam search support (branching with args.use_beam) ---
        if args.use_beam:
            all_beams = beam_search(
                model, src, dec_ipt,
                pad_idx=pad_idx, eos_idx=eos_idx,
                beam_size=args.beam_size, max_len=700,
                src_key_padding_mask=src_key_padding_mask,
                num_return_sequences=args.num_return_sequences,
                num_beam_groups=args.num_beam_groups,
                diversity_strength=args.diversity_strength, 
                temperature=args.temperature,
                sampling_k=args.sampling_k,
                sampling_steps=args.sampling_steps
            )

            
            
            # For each sample and each beam, add to predicted_sequences, source_sequences, target_sequences
            for i, beam_list in enumerate(all_beams):
                src_seq = alignment.extract_sequences(src[i:i+1,1:])[0]
                tgt_seq = alignment.extract_sequences(tgt[i:i+1,1:])[0]
                src_id  = src_ids_batch[i]
                tgt_id  = tgt_ids_batch[i]
                
                for seq in beam_list:
                    cleaned = alignment.extract_sequences(seq.unsqueeze(0)[:,1:])  # List[List[int]]
                    predicted_sequences += cleaned
                    source_sequences.append(src_seq)
                    target_sequences.append(tgt_seq)
                    source_ids.append(src_id)
                    target_ids.append(tgt_id) 

        else:
            # original Greedy generation loop
            for _ in range(700):
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # output_codon_logits = model(dec_ipt, src, src_key_padding_mask=src_key_padding_mask)
                    output_codon_logits = model(dec_ipt, src)
                    
                    output_codon_probs = F.softmax(output_codon_logits, dim=2)
                    next_item = torch.argmax(output_codon_probs, dim=2)[:, -1].unsqueeze(1)
                dec_ipt = torch.cat((dec_ipt, next_item), dim=1)
                if (dec_ipt == eos_idx).any(dim=1).all():
                    break

            source_sequences += alignment.extract_sequences(src[:,1:])
            target_sequences += alignment.extract_sequences(tgt[:,1:])
            predicted_sequences += alignment.extract_sequences(dec_ipt[:,1:])
            source_ids += list(src_ids_batch)
            target_ids += list(tgt_ids_batch)

    # Remove gaps
    predicted_sequences = [[item for item in row if item != dataset.vocab['---']] for row in predicted_sequences]
    predicted_sequences_all =  [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in predicted_sequences]
    source_sequences_dna = [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in source_sequences]
    target_sequences_dna = [''.join([dataset.vocab.index_to_token[c] for c in seq]) for seq in target_sequences]
    predicted_protein_sequences_all = [str(Seq(dna).translate(to_stop=True)) for dna in predicted_sequences_all]
    source_protein_sequences = [str(Seq(dna).translate(to_stop=True)) for dna in source_sequences_dna]
    target_protein_sequences = [str(Seq(dna).translate(to_stop=True)) for dna in target_sequences_dna]
    
    
    
    
    filled_dna_records = []
    filled_protein_records = []
    
    for i, (src_id, tgt_id, src_dna, tgt_dna, pred_dna,
            src_aa, tgt_aa, pred_aa) in enumerate(zip(
                source_ids, target_ids,
                source_sequences_dna, target_sequences_dna, predicted_sequences_all,
                source_protein_sequences, target_protein_sequences, predicted_protein_sequences_all)):
    
        has_original_target = (not args.edition_fasta and tgt_dna)
    
        # --- ID naming (for output) ---
        if args.use_beam:
            beam_idx = (i % args.beam_size) + 1
            pred_id = f"{tgt_id}_predicted_beam{beam_idx}" if has_original_target else f"{tgt_id}_beam{beam_idx}"
        else:
            pred_id = f"{tgt_id}_predicted" if has_original_target else tgt_id
    
        # --- DNA FASTA output ---
        filled_dna_records.append(SeqRecord(Seq(pred_dna), id=pred_id, description=""))
        # filled_dna_records.append(SeqRecord(Seq(src_dna),  id=src_id, description=""))
        # if has_original_target:
        #     filled_dna_records.append(SeqRecord(Seq(tgt_dna), id=tgt_id, description=""))
    
        # --- Protein FASTA output ---
        filled_protein_records.append(SeqRecord(Seq(pred_aa), id=pred_id, description=""))
        filled_protein_records.append(SeqRecord(Seq(src_aa),  id=src_id, description=""))
        if has_original_target:
            filled_protein_records.append(SeqRecord(Seq(tgt_aa), id=tgt_id, description=""))
    
    # --- saving ---
    filled_dna_path = os.path.join(result_folder, "completed_sequences.fasta")
    SeqIO.write(filled_dna_records, filled_dna_path, "fasta")
    print(f"Filled DNA FASTA saved to {filled_dna_path}")
    
    filled_protein_path = os.path.join(result_folder, "completed_proteins.fasta")
    SeqIO.write(filled_protein_records, filled_protein_path, "fasta")
    print(f"Filled protein FASTA saved to {filled_protein_path}")

    
    
    if args.plot:
        # Codon alignment
        plot_obj, score_ratio, src_tgt_mean, tgt_pred_mean = alignment.plot_alignment_scores(source_sequences, target_sequences, predicted_sequences)
        save_with_unique_filename(result_folder, "align.png", plot_obj)
    
        with open(os.path.join(result_folder, "codon_means.txt"), "w") as f:
            f.write(f"src-tgt mean: {src_tgt_mean:.6f}\n")
            f.write(f"tgt-pred mean: {tgt_pred_mean:.6f}\n")
    
        # amino acid alignment
        plot_obj_aa, score_ratio_aa, src_tgt_mean_aa, tgt_pred_mean_aa = alignment.plot_alignment_scores_aa(source_protein_sequences, target_protein_sequences, predicted_protein_sequences_all)
        save_with_unique_filename(result_folder, "align_aa.png", plot_obj_aa)
    
        with open(os.path.join(result_folder, "aa_means.txt"), "w") as f:
            f.write(f"src-tgt mean: {src_tgt_mean_aa:.6f}\n")
            f.write(f"tgt-pred mean: {tgt_pred_mean_aa:.6f}\n")


    print("GC ratio " +  str(CG_ratio(predicted_sequences_all)))
    # print('\n'.join(predicted_sequences_all))
    


    if args.mcts:
        mcts(model, test_loader, dataset.vocab, device, args.edition_fasta, predicted_sequences, result_folder)
