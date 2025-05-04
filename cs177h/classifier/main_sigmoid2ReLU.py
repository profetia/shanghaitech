# coding: utf-8
import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
from tqdm import tqdm
from tensorboardX import SummaryWriter

import typing
import data
import model
import numpy as np
from data import ClassifierDataset
from model import Classifier
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Transformer Model')
parser.add_argument('--data', type=str, default='/home/zhaoqch1/dataset',
                    help='location of the data sequence')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=2000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=5000,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--name', type=str, default='debug',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--seq_len', type=int, required=False, help='length of the sequence to train')

args = parser.parse_args()

PROJECT_NAME = args.name
MODEL_SAVE_PATH = os.path.join('runs', PROJECT_NAME)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Build the model
###############################################################################

if args.seq_len is None:
    args.seq_len = args.bptt//args.batch_size

ntokens = 5
model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.seq_len, args.dropout, is_classify=True)
classifier_model = Classifier(args.seq_len, args.emsize)
model.load_state_dict(torch.load('./fulldata_lr_1e-3_5000_20_3layers.pth'), strict=False)
model.to(device)
classifier_model = classifier_model.to(device)
criterion = nn.CrossEntropyLoss()

###############################################################################
# Load data
###############################################################################

sequence = data.Sequence(args.data)
train_dataset = ClassifierDataset(sequence.train, args.seq_len)
val_dataset = ClassifierDataset(sequence.valid, args.seq_len)
test_dataset = ClassifierDataset(sequence.test, args.seq_len)

test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    data = source[i:i+args.bptt]
    target = source[i:i+args.bptt]
    data=torch.reshape(data,(args.seq_len,args.batch_size))
    return data.to(device), target.to(device)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = 5
    total_length = np.sum([len(x) for x in data_source])
    with torch.no_grad():
        for i in range(0, data_source.size):
            for j in range(0,data_source[i].shape[0]-1, args.bptt):
                if data_source[i].shape[0]-1-j < args.bptt:
                    break
                data, targets = get_batch(data_source[i], j)
                output = model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / total_length

def evaluate_classifier():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    val_dataset.regenerate_data()
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    pbar = tqdm(enumerate(val_data),total=len(val_data))
    with torch.no_grad():
        for index, (input_seq, label) in pbar:
            input_seq = input_seq.to(device)
            label = label.to(device)
            output = model(input_seq)
            output = classifier_model(output)
            total_loss += criterion(output, label).item()
    return total_loss / len(val_data)

def train_classifier(writer, optimizer: torch.optim.Optimizer):
    # Turn on training mode which enables dropout.
    classifier_model.train()
    train_dataset.regenerate_data()
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    total_loss = 0.
    start_time = time.time()
    pbar = tqdm(enumerate(train_data),total=len(train_data))
    for index, (input_seq, label) in pbar:
        input_seq = input_seq.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(input_seq)

        output.requires_grad_(True)
        output = classifier_model(output)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(optimizer.param_groups[0]['params'][0].grad.max() * 1000000)

        total_loss += loss.item()

        if index % args.log_interval == 0 and index > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            pbar.set_description('| lr {:02.8f} | ms/batch {:5.2f} | '
                    'loss {:5.8f}'.format(
                optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval,
                cur_loss))
            total_loss = 0
            global global_iteration
            global_iteration+=1
            writer.add_scalar("loss/train", loss.item(),global_iteration)
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
global_iteration=0
best_val_loss = None
writer = SummaryWriter(logdir=f"../checkpoints/{PROJECT_NAME}")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,classifier_model.parameters()), lr=args.lr)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_classifier(writer, optimizer)
        val_loss = evaluate_classifier()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        writer.add_scalar("loss/epoch", val_loss,epoch)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(MODEL_SAVE_PATH, 'model.pt'), 'wb') as f:
                torch.save(model, f)
            with open(os.path.join(MODEL_SAVE_PATH, 'classifier_model.pt'), 'wb') as f:
                torch.save(classifier_model, f)
            best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join(MODEL_SAVE_PATH, 'model.pt'), 'rb') as f:
    model = torch.load(f)

with open(os.path.join(MODEL_SAVE_PATH, 'classifier_model.pt'), 'rb') as f:
    classifier_model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.

# Run on test data.
test_loss = evaluate_classifier(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
