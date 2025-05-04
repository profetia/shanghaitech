import os
from io import open
import sys
import torch
from Bio import SeqIO
import numpy as np
from typing import Any
import glob
import numpy as np
import re
import glob
import joblib


def get_kmers(sequence, size):
    return " ".join([sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)])

def thread_one_record(rid, record, k, verbose):
    kmer = get_kmers(str(record.seq), k)
    if verbose:
        print(f'Record {rid} done')
    return kmer

def process_one_file(filename, k, verbose):
    records = list(SeqIO.parse(filename, "fasta"))
    with open(f"{filename}_new_{k}.txt", "w") as output:
        output_records = joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(thread_one_record)(i, record, k, verbose) for i, record in enumerate(records))
        output.write("\n".join(output_records))
        print(f'File {filename} done')

path = sys.argv[1]
k = int(sys.argv[2])
verbose = int(sys.argv[3])
print(f'Processing files in {path}')
files = [*sorted(glob.glob(os.path.join(path, "*.fasta"))), *sorted(glob.glob(os.path.join(path, "*.fna")))]
print(f'Found {len(files)} files')
joblib.Parallel(n_jobs=-1)(joblib.delayed(process_one_file)(filename, k, verbose) for filename in files)
