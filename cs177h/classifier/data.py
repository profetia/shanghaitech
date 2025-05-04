import os
from io import open
import torch
from Bio import SeqIO
import numpy as np
from typing import Any
import glob
import numpy as np
import re
import csv
import itertools
# import pickle
import _pickle as pickle
import gc
from torch.utils import data
import random
import multiprocessing
import threading

def handle_record(record, labels):
    # convert the sequence to one-hot encoding
    enc = ordinal_encoder(str(record.seq))
    ncbi_id = record.name.split('|')[-2]
    if ncbi_id not in labels.keys():
        raise ValueError(f'The label of {record.name} is not found')
    return (torch.tensor(enc), labels[ncbi_id])

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


        if os.path.exists('data.pkl'):
            with open('data.pkl', 'rb') as f:
                gc.disable()
                self.data = pickle.load(f)
                gc.enable()
            # self.data = list(map(lambda x: x.float().requires_grad_(True), self.data))
        else:
            self.labels = self.read_label(os.path.join(path, 'source/simulation_abundance_viral50%.CSV'))
            data_list = sorted(list(glob.glob(os.path.join(path, 'simulation_abundance/*.fasta'))))
            # data_list = sorted(list(glob.glob(os.path.join(path, 'virus/2015-/virus2015-.fasta'))))

            # Multithread processing
            self.data = []
            self.threads = []
            for data_item in data_list:
                tokenize_thread = threading.Thread(target=self.tokenize, args=(data_item,))
                tokenize_thread.start()
                self.threads.append(tokenize_thread)
            
            for thread in self.threads:
                thread.join()

            print(f"Gathered {len(self.data)} data. Dumping...")

            # Dump the data
            with open('data.pkl', 'wb') as f:
                pickle.dump(self.data, f)

        random.shuffle(self.data)
        
        train_num = int(len(self.data) * 0.8)
        test_num = int((len(self.data) - train_num) * 0.5)
        valid_num = len(self.data) - train_num - test_num

        # Combine the data and labels
        self.train = self.data[:train_num]
        self.test = self.data[train_num:train_num + test_num]
        self.valid = self.data[train_num + test_num:]

        # self.valid = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))
        # self.test = self.tokenize(os.path.join(path, 'virus/2015-/virus2015-.fasta'))
        # self.train = self.tokenize(os.path.join(path, 'virus/2014-2015/virus2014-2015.fasta'))

    def read_label(self, path):
        """
        Reads the label file.

        CSV in the format of:
        NCBI Accession,type,proportion,number of bp
        
        The label is the indicator of the virus.

        """
        assert os.path.exists(path)

        def str_to_label(label):
            return (float(label == 'virus'))

        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            labels = {}
            for row in reader:
                label = (str_to_label(row[1]), float(row[2]))
                labels[row[0]] = label
        return labels

    
    def tokenize(self, path):
        """Tokenizes a fasta file."""
        print(path)
        assert os.path.exists(path)
        # read the fasta file 
        records = list(SeqIO.parse(path, "fasta"))
        # convert records to one-hot encoding
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

        # Multithread processing
        pool = multiprocessing.Pool(processes=16)
        async_results = [pool.apply_async(handle_record, (records[i], self.labels)) for i in range(len(records))]
        [async_result.wait() for async_result in async_results]
        self.data.extend([async_result.get() for async_result in async_results])
        pool.close()
        

class ClassifierDataset(data.Dataset):

    def __init__(self, data_list, seq_len, sample_num = 12000) -> None:
        super().__init__()
        self.data_list = data_list
        self.data = []
        self.seq_len = seq_len
        self.sample_num = sample_num
        self.regenerate_data()
    
    def regenerate_data(self):
        self.data = []
        for seq, (label, prop) in self.data_list:
            # Randomly sample a slice of the sequence
            if len(seq) > self.seq_len:
                picked_num = np.random.binomial(self.sample_num, prop)
                for i in range(picked_num):
                    start = np.random.randint(0, len(seq) - self.seq_len)
                    end = start + self.seq_len
                    self.data.append((seq[start:end], torch.tensor(label, dtype=torch.float32)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index][0:2]