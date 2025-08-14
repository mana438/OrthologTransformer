import os
import re
from torch.utils.data import Dataset, DataLoader, random_split
import random
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio import SeqIO
import time
import json


class Vocab:
    def __init__(self, tokens, json_path):
        self.json_path = json_path
        self.token_to_index = {}
        self.index_to_token = {}
        self.load_json()  # Load data from JSON file
        self.build_vocab(tokens)

    def load_json(self):
        if os.path.exists(self.json_path) and os.path.getsize(self.json_path) > 0:
            
            try:
                with open(self.json_path, 'r') as file:
                    data = json.load(file)
                    self.token_to_index = data.get('token_to_index', {})
                    self.index_to_token = {int(key): val for key, val in data.get('index_to_token', {}).items()}
            except json.JSONDecodeError:
                # If the JSON file is empty or invalid, initialize
                self.token_to_index = {}
                self.index_to_token = {}
        else:
            # If the file does not exist, initialize
            self.token_to_index = {}
            self.index_to_token = {}


    def save_json(self):
        data = {
            'token_to_index': self.token_to_index,
            'index_to_token': {str(key): val for key, val in self.index_to_token.items()}
        }
        with open(self.json_path, 'w') as file:
            json.dump(data, file)

    def build_vocab(self, tokens):
        for token in tokens:
            if token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token
                self.save_json()  # Update the JSON file every time a new token is added

    def __len__(self):
        return len(self.token_to_index)
    
    def __getitem__(self, token):
        return self.token_to_index[token]

class OrthologDataset(Dataset):
    def __init__(self, OMA_species, json_path):
        self.data = []
        self.ortholog_groups = set()
        # Get the list of all species names
        with open(OMA_species, 'r') as file:
            species_names = file.read().splitlines()
        
        # List of all codons
        codons = [f"{x}{y}{z}" for x in "ACGT" for y in "ACGT" for z in "ACGT"]
        gap = ["---"]

        # Special tokens
        spc = ['<pad>', '<bos>', '<eos>']

        # Create a token list containing both codons and species names
        tokens =  spc + codons + gap + species_names
        self.vocab = Vocab(tokens, json_path)
        self.vocab_target = spc + codons + gap


    def load_data(self, ortholog_files, reverse):
        dataset = []
        
        # Read each ortholog relationship file
        for ortholog_file in ortholog_files:
            # Default value
            group_number = "NA"
            # Pattern 1: group number present
            match = re.search(r"group(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_\d+\.fasta", ortholog_file)

            if match:
                group_number = match.group(1)
                species_name_1 = match.group(2)
                species_name_2 = match.group(3)
            else:
                # Pattern 2: no group number → assume format like species1_species2.fasta
                # Example: ECOLI_IDESA.fasta, ECOLI_IDESA_1.fasta
                basename = os.path.basename(ortholog_file)
                fallback_match = re.search(r"([A-Z0-9]+)_([A-Z0-9]+)(?:_\d+)?\.fasta", basename)
                if fallback_match:
                    species_name_1 = fallback_match.group(1)
                    species_name_2 = fallback_match.group(2)
                else:
                    print(f"Filename pattern not recognized: {ortholog_file}")
                    continue

            sequences = []
            ids = []
            # Read the fasta file and store each sequence in a dictionary
            for seq_record in SeqIO.parse(ortholog_file, "fasta"):
                sequences.append(str(seq_record.seq))
                ids.append(seq_record.id)

            # If the pair is valid, add to the dataset
            if self.is_valid_pair(sequences[0], sequences[1]):
                seq1_codons = self.convert_to_codons(sequences[0], species_name_1)                            
                seq2_codons = self.convert_to_codons(sequences[1], species_name_2)                            
                 # Add base data
                dataset.append((group_number, seq1_codons, seq2_codons, ids[0], ids[1]))
                if reverse:
                    dataset.append((group_number, seq2_codons, seq1_codons, ids[1], ids[0]))
        
        return dataset


    def convert_to_codons(self, seq, seq_species):
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        seq_species_index = self.vocab[seq_species]
        codon_seq = [seq_species_index] +  [self.vocab['<bos>']] + [self.vocab[codon] for codon in codons] + [self.vocab['<eos>']]
        return codon_seq


    # Determine if the sequence pair is valid
    def is_valid_pair(self, seq1, seq2):
        length1, length2 = len(seq1), len(seq2)
        valid_chars = set("ATGC-")

        return (
            length1 <= 2100
            and length2 <= 2100
            and length1 % 3 == 0  # Added condition: length1 divisible by 3
            and length2 % 3 == 0  # Added condition: length2 divisible by 3
            and set(seq1.upper()) <= valid_chars
            and set(seq2.upper()) <= valid_chars
        )

    def len_vocab_input(self):
        return len(self.vocab)

    def len_vocab_target(self):
        return len(self.vocab_target)

    # Return the data at the specified index
    def __getitem__(self, index):
        return self.data[index]

    # Return the size of the dataset
    def __len__(self):
        return len(self.data)
