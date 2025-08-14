import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, distributed
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from util import custom_collate_fn, alignment, count_nonzero_matches, load_params, check_condition, allreduce, count_parameters, CG_ratio
from mcts import mcts
from model import CodonTransformer
from read import OrthologDataset

import os
import time
import sys

import traceback
import random

import json
import argparse
import glob
from arg_parser import get_args
import pickle
import schedulefree

args = get_args()

# Result file
result_folder = args.result_folder
# Change standard output destination
sys.stdout = open(os.path.join(result_folder, "output.txt"), "a")
# Output parameters
with open(os.path.join(result_folder, "args.json"), "a") as f:
    json.dump(vars(args), f, indent=4)

script_dir = os.path.dirname(os.path.abspath(__file__))
default_species_path = os.path.join(script_dir, "prokaryotes_group.txt")
default_vocab_path = os.path.join(script_dir, "vocab_OMA.json")

# Create an OrthologDataset object
dataset = OrthologDataset(default_species_path,default_vocab_path)

start_time = time.time() 
# If pickle file exists, load from it
if args.pickle_path:
    with open(args.pickle_path, "rb") as f:
        train_dataset = pickle.load(f)
else:
    # List of ortholog relationship files for training
    ortholog_files_train = glob.glob(args.ortholog_files_train.replace("'", ""))
    print("pre: ",len(ortholog_files_train), flush=True)
    train_dataset = dataset.load_data(ortholog_files_train, args.reverse)

print(f"data load Time: {time.time() - start_time:.2f} seconds", flush=True)
print("train: ", len(train_dataset), flush=True)


# Count unique DNA sequences
unique_sequences = set()
for _, seq1_codons, seq2_codons, _ in train_dataset:
    # Convert codon indices back to sequence representation for uniqueness check
    seq1 = tuple(seq1_codons[2:-1])  # Skip species index, <bos> and <eos>
    seq2 = tuple(seq2_codons[2:-1])  # Skip species index, <bos> and <eos>
    unique_sequences.add(seq1)
    unique_sequences.add(seq2)
print(f"Number of unique DNA sequences: {len(unique_sequences)}", flush=True)    

if args.horovod:
    import horovod.torch as hvd
    hvd.init()
    # DataLoader for Horovod
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, collate_fn=custom_collate_fn)
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device("cuda", hvd.local_rank())
    print(f"Process {hvd.rank()}: Local rank {hvd.local_rank()}", flush=True)

else:
    hvd = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

# token size
vocab_size_target = dataset.len_vocab_target()
vocab_size_input = dataset.len_vocab_input()

# Create a model instance
model = CodonTransformer(vocab_size_target, vocab_size_input, args.d_model, args.nhead, args.num_encoder_layers, args.num_decoder_layers, args.dim_feedforward, args.dropout).to(device)
# model = torch.compile(model, backend="aot_eager")

if args.model_input:
    if args.horovod:
        # In Horovod environment, map to GPU assigned to each process
        weights = torch.load(args.model_input, map_location=lambda storage, loc: storage.cuda(hvd.local_rank()))
    else:
        # For single GPU or CPU mapping, specify like below
        # weights = torch.load(args.model_input, map_location='cpu')
        weights = torch.load(args.model_input, map_location='cuda:0')
    model.load_state_dict(weights, strict=True)
    
# if args.model_input:
#     weights = torch.load(args.model_input)
#     model.load_state_dict(weights, strict=False)
print(f'Total number of parameters: {count_parameters(model)}', flush=True)



# model = torch.compile(model)

# Loss function and optimization algorithm
pad_idx = dataset.vocab['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
# optimizer = optim.NAdam(model.parameters(), lr=args.learning_rate)
# optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
# optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.learning_rate)
# optimizer.train() #schedulefree

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize GradScaler
scaler = GradScaler()

if args.horovod:
    # Horovod-specific optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # Horovod: Broadcast model parameters and optimizer state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)


# Training
for epoch in range(args.num_epochs):
    start_time = time.time()  # Record start time of the epoch
    try:
        print(epoch, " epoch start", flush=True)
        # Display number of samples handled by each GPU
        if hasattr(train_loader.sampler, 'num_samples') and args.horovod:
            print(f"Process {hvd.rank()} is handling {train_loader.sampler.num_samples} samples.", flush=True)

        model.train()
        
        train_correct_predictions = 0
        train_total_predictions = 0

        total_loss = 0
        num_batches = 0

        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()  # Batch processing start time
            
            tgt = batch[1].to(device)                
            src = batch[2].to(device)

            dec_ipt = tgt[:, :-1]
            dec_tgt = tgt[:, 1:] # Shift right by one
            
            # Create padding masks
            src_key_padding_mask = (src == pad_idx).to(device)
            tgt_key_padding_mask = (dec_ipt == pad_idx).to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_codon = model(dec_ipt, src, None, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                output_codon = output_codon.transpose(1, 2) # for CrossEntropyLoss
                loss = criterion(output_codon, dec_tgt)

            # Initialize gradients
            optimizer.zero_grad()
            
            # Scale gradients and backpropagate
            scaler.scale(loss).backward()
            
            # Horovod synchronization
            if args.horovod:
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)   # Update with scaled gradients
            else:
                scaler.step(optimizer)
            
            # Update scaling factor
            scaler.update()
            
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate number of correct predictions and total predictions (for codons)
            output_codon = output_codon.transpose(1, 2)
            output_codon = torch.argmax(output_codon, dim=2)
            match_count, all_count = count_nonzero_matches(tgt[:, 1:], output_codon)
            train_correct_predictions += match_count
            train_total_predictions += all_count
                
            batch_time = time.time() - batch_start_time


        # if (epoch + 1) == args.num_epochs:
        train_accuracy = train_correct_predictions / train_total_predictions
        print(f"Training Accuracy: {train_accuracy:.4f}", flush=True)
        
        print(f"Epoch {epoch + 1} Time: {time.time() - start_time:.2f} seconds, Avg Batch Time: {(time.time() - start_time) / num_batches:.2f} seconds", flush=True)

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}", flush=True)
        # save model
        torch.save(model.state_dict(), os.path.join(result_folder, os.path.basename("model_weights.pth")))
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error occurred: {e}\n{tb}", flush=True)
        print(f"Skipping this epoch and moving to next epoch.", flush=True)
        continue