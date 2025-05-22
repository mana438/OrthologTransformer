#!/usr/bin/env python3
"""
This script reads DNA sequences from a FASTA file, translates them to amino acid sequences,
performs pairwise global alignment between the first sequence and each subsequent sequence
using the BLOSUM62 scoring matrix, and reports the counts of insertions, deletions,
synonymous substitutions, and nonsynonymous substitutions.
"""
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import pairwise2


def get_codon_list(dna_seq):
    """Split a DNA sequence into codons (3 nucleotides each)."""
    seq_str = str(dna_seq)
    length = (len(seq_str) // 3) * 3
    return [seq_str[i:i+3] for i in range(0, length, 3)]


def analyze_pairwise(ref_codons, ref_prot, query_codons, query_prot, matrix):
    """
    Align two protein sequences and count insertion, deletion,
    synonymous, and nonsynonymous differences based on underlying codons.
    """
    # Perform global alignment using BLOSUM matrix
    aln_ref, aln_query, score, *_ = pairwise2.align.globaldx(ref_prot, query_prot, matrix, one_alignment_only=True)[0]

    idx_ref = idx_query = 0
    insertions = deletions = synonymous = nonsynonymous = 0

    for aa_ref, aa_query in zip(aln_ref, aln_query):
        if aa_ref == '-':
            insertions += 1
            idx_query += 1
        elif aa_query == '-':
            deletions += 1
            idx_ref += 1
        else:
            cod_ref = ref_codons[idx_ref]
            cod_query = query_codons[idx_query]
            if aa_ref == aa_query:
                # Same amino acid but different codon
                if cod_ref != cod_query:
                    synonymous += 1
            else:
                nonsynonymous += 1
            idx_ref += 1
            idx_query += 1

    return insertions, deletions, synonymous, nonsynonymous, score


def main():
    parser = argparse.ArgumentParser(description="DNA-to-AA alignment analysis")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file of DNA sequences")
    args = parser.parse_args()

    # Read all DNA records
    records = list(SeqIO.parse(args.input, "fasta"))
    if len(records) < 2:
        parser.error("Need at least two sequences in the FASTA file.")

    # Prepare reference sequence
    ref = records[0]
    ref_codons = get_codon_list(ref.seq)
    ref_prot = ref.seq.translate()

    matrix = substitution_matrices.load("BLOSUM62")


     # Analyze each subsequent record
    for rec in records[1:]:
        query_codons = get_codon_list(rec.seq)
        query_prot   = rec.seq.translate()
        ins, del_, syn, nonsyn, score = analyze_pairwise(
         ref_codons, ref_prot,
         query_codons, query_prot,
         matrix
        )
        # カウントを(Ins/ Del/ Syn/ Nonsyn)形式で出力
        print(f"{rec.id}\t({ins}/ {del_}/ {syn}/ {nonsyn})\t{score:.2f}")


if __name__ == "__main__":
    main()
