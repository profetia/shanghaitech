import argparse
import glob
import os
import random
import pandas
from Bio import SeqIO
import joblib

arg_parser = argparse.ArgumentParser(description='sample tsv')
arg_parser.add_argument('--file-path', type=str, help='Path to file')
arg_parser.add_argument('--sample-size', type=int, help='Sample size', default=250)
arg_parser.add_argument('--stride', type=int, help='Stride', default=50)
arg_parser.add_argument('--verbose', type=bool, help='Verbose', default=True)
arg_parser.add_argument('--policy', type=str, help='Policy', default='sequential')

args = arg_parser.parse_args()

def sample_one_record(record_id, record, stride, sample_size, policy):
    raw_record, label = record.split('\t')
    raw_record = raw_record.split(' ')
    if policy == 'random':        
        target_num = len(raw_record) // (sample_size - stride)
        target_num = target_num // 10
        result = []
        for _ in range(target_num):
            start = random.randint(0, len(raw_record) - sample_size)
            result.append(f'{" ".join(raw_record[start:start+sample_size])}\t{label}')
    else:
        result = [ f'{" ".join(raw_record[i:i+sample_size])}\t{label}' 
            for i in range(0, len(raw_record), stride) ]

    if args.verbose:
        print(f'Sample {record_id} Finished')

    return result

def process_one_file(filename):
    inputs = []
    with open(filename, "r") as f:
        inputs = f.read().split('\n')
    outputs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(sample_one_record)(i, record, args.stride, args.sample_size, args.policy) for i, record in enumerate(inputs))
    outputs = [item for sublist in outputs for item in sublist]
    with open(f"{filename}_sampled.txt", "w") as output:
        output.write("\n".join(outputs))
        print(f'File {filename} done')

process_one_file(args.file_path)