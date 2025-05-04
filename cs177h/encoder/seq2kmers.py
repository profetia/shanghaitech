import os
from io import open
import torch
from Bio import SeqIO
import numpy as np
from typing import Any
import glob
import numpy as np
import re
import glob

def getKmers(sequence, size):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
path = '../dataset/human/'
for filename in sorted(glob.glob(os.path.join(path, "*.fna"))):
    records = list(SeqIO.parse(filename, "fasta"))
    with open("../dataset/human/GRCh38_latest_genomic_5mer.txt", "a") as output:
        for i in range(len(records)):
            kmer=getKmers(str(records[i].seq), 5)
            #write the kmer to a txt file after the end of the file
            output.write(str(kmer))
            output.write('\n')
            print("Sequence"+str(i)+"/"+str(len(records))+" done")

