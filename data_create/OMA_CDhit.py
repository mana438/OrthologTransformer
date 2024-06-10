import subprocess
import os
import sys
from collections import defaultdict
from Bio import SeqIO

def process_group(group_id, seq_ids, seq_data, cdhit_path):
    fasta_file = f'/home/aca10223gf/workplace/sample/group{group_id}.fa'
    clstr_file = f'/home/aca10223gf/workplace/sample/group{group_id}'

    with open(fasta_file, 'w') as f:
        for seq_id in seq_ids:
            if seq_id in seq_data:
                record = seq_data[seq_id]
                SeqIO.write(record, f, 'fasta')

    threads = os.cpu_count()
    memory = 100000  # 100GB
    cmd = f'{cdhit_path} -i {fasta_file} -o {clstr_file} -c 0.8 -M {memory} -T {threads}'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    rep_seqs = []
    with open(clstr_file + ".clstr", 'r') as f:
        for line in f:
            if line.startswith('>'):
                continue
            fields = line.strip().split()
            if fields[-1] == '*':
                rep_seq = fields[-2].lstrip('>').rstrip('...')
                rep_seqs.append(rep_seq)

    os.remove(fasta_file)
    os.remove(clstr_file)
    os.remove(clstr_file + ".clstr")
    
    return group_id, rep_seqs

if __name__ == '__main__':
    group_file = sys.argv[1]

    # CD-HITのパスを設定
    cdhit_path = '/home/aca10223gf/workplace/tools/cdhit/cd-hit-est'

    # prokaryotes.cdna.faの読み込み
    seq_data = {}
    for record in SeqIO.parse('/home/aca10223gf/workplace/data/OMA_database/prokaryotes.cdna.fa', 'fasta'):
        if len(record.seq) < 2100:
            seq_data[record.id] = record

    if seq_data:
        # グループファイルの読み込みとクラスタリングの実行
        with open(f'/home/aca10223gf/workplace/data/OMA_database/oma-groups-split/{group_file}', 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                group_id = fields[0]
                seq_ids = fields[2:]
                group_id, rep_seqs = process_group(group_id, seq_ids, seq_data, cdhit_path)
                
                # 結果の出力
                if rep_seqs:
                    output_file = f'/home/aca10223gf/workplace/data/OMA_database/cdhit_database/oma-groups.cdhit.{group_id}.txt'
                    with open(output_file, 'w') as out_f:
                        out_f.write(f'{group_id}\t' + '\t'.join(rep_seqs) + '\n')