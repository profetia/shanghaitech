import os
from io import open
import torch
from Bio import SeqIO
import numpy as np
from typing import Any
import glob
import numpy as np
import re
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','t','z','c','g']))
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(string_to_array(my_array))
    return integer_encoded
class Sequence(object):
    def __init__(self, path):
        # train_list=[]
        # train_list.append(glob.glob(r'../dataset/virus/-2014/*.py'))
        self.extradata = self.valid = self.tokenize(os.path.join(path, 'human/GRCh38_latest_genomic.fna'))
        self.train = self.tokenize(os.path.join(path, 'prokaryote/-2014/prokaryote-2014_1-3000.fasta'))
        # for file in ['simulation_abundance/sa_1001-2000.fasta','simulation_abundance/sa_2001-3000.fasta',\
        #     'simulation_abundance/sa_3001-4000.fasta','simulation_abundance/sa_4001-5000.fasta'\
        #     'human/GRCh38_latest_genomic.fna']:
        for file in [\
            'prokaryote/-2014/prokaryote-2014_3001-6000.fasta',\
            'prokaryote/-2014/prokaryote-2014_6001-9000.fasta',\
            'prokaryote/-2014/prokaryote-2014_9001-12000.fasta',\
            'prokaryote/-2014/prokaryote-2014_12001-16132.fasta'\
            ]:
            np.concatenate((self.train,self.tokenize(os.path.join(path, file))))
            np.concatenate((self.train,self.extradata[:600]))
            np.random.shuffle(self.train)
        # self.valid = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))
        # self.test = self.tokenize(os.path.join(path, 'virus/2015-/virus2015-.fasta'))
        # self.train = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))
        self.valid = self.tokenize(os.path.join(path, 'prokaryote/2014-2015/prokaryote2014-2015.fasta'))
        np.concatenate((self.valid,self.extradata[601:650]))
        np.random.shuffle(self.valid)
        self.test = self.tokenize(os.path.join(path, 'prokaryote/2015-/prokaryote2015-_1-1000.fasta'))
        for file in [\
            'prokaryote/2015-/prokaryote2015-_1001-1500.fasta',\
            'prokaryote/2015-/prokaryote2015-_1501-2000.fasta',\
            'prokaryote/2015-/prokaryote2015-_2001-3000.fasta',\
            'prokaryote/2015-/prokaryote2015-_3001-4000.fasta',\
            'prokaryote/2015-/prokaryote2015-_4001-5000.fasta',\
            'prokaryote/2015-/prokaryote2015-_5001-6248.fasta'\
            ]:
            np.concatenate((self.test,self.tokenize(os.path.join(path, file))))
            np.concatenate((self.test,self.extradata[651:]))
            np.random.shuffle(self.test)
    
    def tokenize(self, path):
        """Tokenizes a fasta file."""
        print(path)
        assert os.path.exists(path)
        # read the fasta file 
        records = list(SeqIO.parse(path, "fasta"))
        # convert records to one-hot encoding
        sequence=[]
        def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGU',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
            """One-hot encode a sequence."""
            def to_uint8(string):
                return np.frombuffer(string.encode('ascii'), dtype=np.uint8)

            hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
            hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
            hash_table[to_uint8(neutral_alphabet)] = neutral_value
            hash_table = hash_table.astype(dtype)
            return hash_table[to_uint8(sequence)]


        for record in records:
            # convert the sequence to one-hot encoding
            enc = ordinal_encoder(str(record.seq))
            sequence.append(torch.tensor(enc))
            

        return np.array(sequence)
