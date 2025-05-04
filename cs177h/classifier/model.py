import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000): # d_model = 200 ~ embedding size
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Classifier(nn.Module):
    """Classify the feature map of the encoder"""

    def __init__(self, seq_length, embedding_size) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        # Original
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(1, 3, 5, 2, 1),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        #     nn.Conv2d(3, 16, 5, 2, 1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(97216, 1024),
        #     nn.BatchNorm1d(1024), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(256, 1)
        # )

        # Simplified
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 3, 5, 2, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(97216, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Only linear
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(100000, 1024),
        #     nn.BatchNorm1d(1024), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )
        
   
    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1)

        # CONV
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        # Only Linear
        # x = x.reshape(-1, x.size(1) * x.size(2))

        x = self.linear_layer(x)
        return x.squeeze()


class ClassifierLinear(nn.Module):
    """Classify the feature map of the encoder"""

    def __init__(self, seq_length, embedding_size) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        # Original
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(1, 3, 5, 2, 1),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        #     nn.Conv2d(3, 16, 5, 2, 1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(97216, 1024),
        #     nn.BatchNorm1d(1024), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(256, 1)
        # )

        # Simplified
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(1, 3, 5, 2, 1),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        #     nn.Conv2d(3, 16, 5, 2, 1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(97216, 1024),
        #     nn.BatchNorm1d(1024), 
        #     nn.ReLU(),
        #     nn.Dropout(0.2), 
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )

        # Only linear
        self.linear_layer = nn.Sequential(
            nn.Linear(100000, 2048),             
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2), 
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128), 
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
   
    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1)

        # CONV
        # x = x.unsqueeze(1)
        # x = self.conv_layer(x)
        # x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        # Only Linear
        x = x.reshape(-1, x.size(1) * x.size(2))

        x = self.linear_layer(x)
        return x.squeeze()


class ClassifierNew(nn.Module):
    """Classify the feature map of the encoder"""

    def __init__(self, seq_length, embedding_size) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        # Original
        self.conv_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, 5, 2, 2)), # [N, 1, 500, 200] -> [N, 16, 250, 100]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(16, 64, 5, 2, 2)), # [N, 16, 250, 100] -> [N, 64, 125, 50]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 256, 6, 4, 1)), # [N, 64, 125, 50] -> [N, 256, 31, 12]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(256, 1024, 6, 4, 1)), # [N, 256, 31, 12] -> [N, 1024, 7, 3]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),            
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(21504, 2048),
            nn.BatchNorm1d(2048), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512, 1)
        )
   
    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1)

        # CONV
        x = x.unsqueeze(1)
        x = self.conv_layer(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = self.linear_layer(x)
        return x.squeeze()

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, seq_len, dropout=0.5, is_classify=False):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.is_classify = is_classify
        if is_classify:
            # self.clasifier = Classifier(seq_len, ninp)
            self.freeze_encoder()
        else:
            self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.transformer_encoder.requires_grad_(False)

    def forward(self, src, has_mask=True): # src: batch_size x seq_len x 5
        src = src.transpose(0, 1) # src: batch_size x seq_len x 5 -> seq_len x batch_size x 5
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp) # seq_len x batch_size x 5 -> seq_len x batch_size x embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        if self.is_classify:
            # output.requires_grad_(True)
            # return self.clasifier(output)
            return output
        else:
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1)