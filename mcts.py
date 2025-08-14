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
        self.sequence = sequence  # Current codon sequence
        self.dec_ipt = dec_ipt
        self.parent = parent  # Parent node
        self.depth = depth
        self.children = []   # List of child nodes
        self.visits = 0  # Number of visits to this node
        self.value = 0  # Value of this node

    def is_fully_expanded(self):
        return len(self.children) > 0  # Check if the node has been expanded

    def best_child(self):
        # Select the best child node using the UCB1 algorithm
        return max(self.children, key=lambda node: node.value / (node.visits + 1e-6) +
                                        0.5 * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6)))

    def expand(self, model, vocab, src, len_predicted_seq):
        # If the node is not fully expanded, create new child nodes
        if self.visits > 3 and not self.is_fully_expanded() and len(self.sequence) < len_predicted_seq:
            # Get token probabilities
            output_codon = model(self.dec_ipt, src)
            # Apply Softmax to normalize probabilities
            probabilities = F.softmax(output_codon, dim=2)

            # Sort tokens by probability and get the top 3 indices
            top_values, top_indices = torch.topk(probabilities, 3, dim=2)

            for i in range(3):
                # Get the i-th top token
                next_item = top_indices[:, -1, i].unsqueeze(1)
                next_value = top_values[:, -1, i].unsqueeze(1)

                # Output the predicted token and its value
                predicted_token = vocab.index_to_token[next_item[0].item()]
                # print(f"Top {i+1} predicted token: {predicted_token}, value: {next_value[0].item()}")
                
                if next_value[0].item() > 0.1:
                    # Add the predicted token to the decoder input
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
    # Calculate reward based on the final sequence's GC content
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
    ss, mfe = RNA.fold(sequence)
    
    gc_content_reward = 20/100**abs(gc_content - target_gc_content)
    mfe_reward = abs(mfe)/20
    # all_reward = gc_content_reward + mfe_reward
    all_reward = gc_content_reward
    return all_reward, mfe, gc_content

def simulate(node, model, src, template_seq, vocab, epoch):
    # Rollout phase: randomly select actions and calculate reward at the final state
    
    if (epoch % 500) == 0:
        sequence = node.sequence.copy()  # Copy the current sequence
        dec_ipt = node.dec_ipt.detach()
        for i in range(1000):
            output_codon = model(dec_ipt, src)
            output_codon = torch.argmax(output_codon, dim=2)
            # Get the most probable codon token
            next_item = output_codon[:, -1].unsqueeze(1) 
            # Add the predicted codon token to the decoder input
            dec_ipt = torch.cat((dec_ipt, next_item), dim=1)

            sequence.append(vocab.index_to_token[next_item[0].item()])

            # End if an end-of-sentence token is output
            # Determine if '<eos>' is included in each line
            end_token_count = ((dec_ipt == vocab['<eos>']) | (dec_ipt == vocab['<pad>'])).any(dim=1).sum().item()
            if end_token_count == len(src):
                break
    else:
        sequence = node.sequence.copy()  # Copy the current sequence
        sequence = sequence + template_seq[len(sequence) : ]

    return sequence  # Return the final sequence for reward calculation

def backpropagate(node, value):
    # Backpropagation: propagate value to parent nodes
    while node:
        node.visits += 1  # Increment visit count
        node.value += value  # Update value
        node = node.parent  # Move to parent node

def main_loop(root, model, vocab, tgt, src, predicted_seq, result_folder, iterations=1000000):
    rewards = []  # List to store reward values
    gc_contents = []  # List to store GC content values
    mfes = []  # List to store Minimum Free Energy (MFE) values
    tree_depths = []  # List to store tree depths

    # Main loop of MCTS
    template_seq = predicted_seq.copy()
    for i in range(iterations):
        node = root  # Start from the root node
        while node.is_fully_expanded():
            node = node.best_child()
        node.expand(model, vocab, src, len(predicted_seq)) # Expand only if node visits exceed a certain threshold
        if node.children:
            child = node.children[0]  # Get the first added child node
        else:
            child = node
        sequence = simulate(child, model, src, template_seq, vocab, i)  # Run simulation for the child node

        if (i % 500) == 0:
            template_seq = sequence.copy()

        all_reward, mfe, gc_content = reward(sequence)
        
        backpropagate(child, all_reward)  # Backpropagate the value

        rewards.append(all_reward)
        gc_contents.append(gc_content)
        mfes.append(mfe)
        tree_depths.append(child.depth)  # Add tree depth to list

        if (i + 1) % 100 == 0:  # Every 100 iterations, plot and save graphs
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
            plt.savefig(os.path.join(result_folder, "mcts_metrics.png"))
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
        
        root = Node(sequence=sequence, dec_ipt=dec_ipt)  # Create initial node
        main_loop(root, model, vocab, tgt, src, predicted_sequences[0], result_folder)   # Run MCTS
        best_child = root.best_child()  # Get best child node
        print(f"Best child sequence: {best_child.sequence}, Value: {best_child.value / best_child.visits}")  # Output results
