import argparse
import glob
import os
import pandas
from Bio import SeqIO
import joblib

arg_parser = argparse.ArgumentParser(description='Convert kmer counts to tsv')
arg_parser.add_argument('--label_path', type=str, help='Path to labels file', default='../dataset/source/simulation_abundance_viral10%.CSV')
arg_parser.add_argument('--fasta_path', type=str, help='Path to fasta file', default='../dataset/simulation_abundance')
arg_parser.add_argument('--verbose', type=bool, help='Verbose', default=True)

args = arg_parser.parse_args()
labels = pandas.read_csv(args.label_path)
print(f'Processing files in {args.fasta_path}')
fasta_files = [*sorted(glob.glob(os.path.join(args.fasta_path, "*.fasta"))), 
    *sorted(glob.glob(os.path.join(args.fasta_path, "*.fna")))]
print(f'Found {len(fasta_files)} files')

def get_label(ncbi_id):
    return int(labels[labels['NCBI Accession'] == ncbi_id]['type'].values[0] == 'virus')

def concat_with_label(kmer, ncbi_id):
    real_ncbi_id = ncbi_id.split('|')[3]
    if real_ncbi_id not in labels['NCBI Accession'].values:
        raise Exception(f'NCBI ID {real_ncbi_id} not found in labels')

    if args.verbose:
        print(f'Label for {real_ncbi_id} is {get_label(real_ncbi_id)}')
    return f'{kmer.upper()}\t{get_label(real_ncbi_id)}'

def process_one_file(filename):
    records = list(SeqIO.parse(filename, "fasta"))

    if args.verbose:
        print(f'Found {len(records)} records in {filename}')

    with open(f"{filename}_new.txt", "r") as kmers_file:
        kmers = kmers_file.read().split('\n')
    
    if args.verbose:
        print(f'Found {len(kmers)} kmers in {filename}')

    with open(f"{filename}_new_labeled.txt", "w") as output:
        output_records = joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(concat_with_label)(kmer, record.id) for kmer, record in zip(kmers, records))

        output.write("\n".join(output_records))
        print(f'File {filename} done')

joblib.Parallel(n_jobs=-1)(joblib.delayed(process_one_file)(filename) for filename in fasta_files)
