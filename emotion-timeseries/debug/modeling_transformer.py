"""Implement transformer model as presented in https://arxiv.org/abs/1706.03762."""

import copy
from collections import OrderedDict
from torch.autograd import Variable
from utils import colabelMask
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiSequential(nn.Sequential):
    """Sequential model with as many inputs as outputs."""

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def clones(module, n):
    """Produce n identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def seq_clones(module, seq_len):
    """Produce sequential model with seq_len identical layers."""

    return MultiSequential(
        OrderedDict(
            [('layer{}'.format(i), copy.deepcopy(module))
             for i in range(seq_len)]
        )
    )


class MultiHeadAttention(nn.Module):
    """Implement multi-head attention module.

    Args:
        model_dim: dimension of embedding.
        nheads: number of attention heads.
        mask: sets whether an input mask should be used.
    """

    def __init__(self, model_dim, nheads, p=0.1, mask=None):
        super(MultiHeadAttention,self).__init__()

        self.mask = mask
        self.nheads = nheads
        self.model_dim = model_dim

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)

        self.att = None

        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        """Compute multi-head attention forward pass.

        Args:
            query: tensor with shape (batch_size, sentence_len1, model_dim).
            key: tensor with shape (batch_size, sentence_len2, model_dim).
            value: tensor with shape (batch_size, sentence_len2, model_dim).

        Returns:
            tensor with shape (batch_size, sentence_len1, model_dim).
        """

        assert self.model_dim % self.nheads == 0

        key_dim = self.model_dim//self.nheads
        shape_q = query.shape[:2]+(self.nheads, key_dim)
        shape_k = key.shape[:2]+(self.nheads, key_dim)
        shape_v = value.shape[:2]+(self.nheads, key_dim)

        print('multihead query.......', query)
        print('multihead key.......', key)
        print('multihead value.......', value)
        print('linear query........', self.linear_q(query))
        print('linear key........', self.linear_q(key))
        print('linear value........', self.linear_q(value))


        ret = self.attention(
            self.linear_q(query).reshape(shape_q),
            self.linear_k(key).reshape(shape_k),
            self.linear_v(value).reshape(shape_v)
        )
        ret = ret.reshape(ret.shape[:2] + (self.model_dim,))

        return self.dropout(self.linear_out(ret))


    def attention(self, query, key, value):
        """Compute scaled dot-product attention.

        Args:
            query: tensor with shape (batch_size, sentence_len1, nheads, key_dim).
            key: tensor with shape (batch_size, sentence_len2, nheads, key_dim).
            value: tensor with shape (batch_size, sentence_len2, nheads, key_dim).

        Returns:
            tensor with shape (batch_size, sentence_len1, nheads, key_dim).
        """
        print('query .......', query)
        print('key .........', key)
        print('value.........', value)
        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
        print('start score ........', score)
        if self.mask == 'triu':
            mask = torch.triu(
                torch.ones(score.shape, dtype=torch.bool), diagonal=1
            )
            score[mask] = -float('inf')

        if self.mask == 'diag':
            mask = torch.eye(
                n=score.shape[2], m=score.shape[3], dtype=torch.float,
            )
            mask = mask.reshape(-1).repeat((1, np.prod(score.shape[:2]))).reshape(score.shape)
            score[mask.type(torch.long)] = -float('inf')

        if self.mask == 'co-label':
            adj_file = 'embedding/positiveEmotion_adj.pkl'
            mask = colabelMask(t=0.2, adj_file=adj_file)
            print('mask....', mask)
            score = score.double()
            print('bafore score.....', score)
            mask = torch.as_tensor(mask, dtype=torch.double).cuda()
            score = torch.mul(score, mask)
            print('after score.....', score)
            score[score == float('inf')] = float('-inf')
            print('score.....', score)
            score = score.double()

        self.att = F.softmax(score / np.sqrt(score.shape[-1]), dim=-1)
        print('self.att.....', self.att)
        ret = torch.einsum('bhqk,bkhd->bqhd', self.att.float(), value)
        print('ret.....', ret)

        return ret


class Embedding(nn.Module):
    """Implement input and output embedding with tied weights."""

    def __init__(self, seq_len, model_dim):
        super(Embedding, self).__init__()

        self.seq_len = seq_len
        self.model_dim = model_dim

        self.encoder = nn.Embedding(seq_len, model_dim)
        self.decoder = nn.Linear(model_dim, seq_len, bias=False)

        self.decoder.weight = self.encoder.weight

    def forward(self, x, inverse=False):
        if inverse:
            return self.decoder(x)

        return self.encoder(x.to(torch.int64)) * np.sqrt(self.model_dim)


class PositionalEncoder(nn.Module):
    """Implement the Positional Encoder function.

    Based on https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, d_model, max_len=150, p=0.1):
        super(PositionalEncoder, self).__init__()
        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.tensor(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model),dtype=torch.float))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pos_enc_o = torch.cat([pos_enc[:, 1::2],torch.zeros(pos_enc[:, 1::2].shape[0],1)], dim=-1)
            pos_enc_o = torch.cos(position * div_term)
            pos_enc[:, 1::2] = pos_enc_o[:,0:-1]
        else:
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pe', pos_enc)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.cuda()
        x = x + Variable(self.pe[:, :x.size(1)].cuda(), requires_grad=False)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Implement encoder sub-layers."""

    def __init__(self, model_dim, hidden_dim, nheads, p=0.1, mask=None):
        super(EncoderLayer, self).__init__()
        if mask == 'co-label':
            # Using the transformer to learn the correlation between labels depend on the label co-occurrence matrix.
            self.mhatt = MultiHeadAttention(
                model_dim, nheads, p, mask='co-label',
            )
        if mask == None:
            # The sequence length of EEG samples is same, so the mask is not need.
            self.mhatt = MultiHeadAttention(
                model_dim, nheads, p, mask=None,
            )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(p=p),
        )

        self.layer_norms = clones(nn.LayerNorm(model_dim), 2)

    def forward(self, src):
        print('src .......', src)
        print('output of MultiheadAttention.......', self.mhatt(src, src, src) )
        src_att = self.layer_norms[0](self.mhatt(src, src, src) + src)
        print('layer_norm output.........', src_att)
        print('output of ffn.........', self.ffn(src_att))
        src_out = self.layer_norms[1](self.ffn(src_att) + src_att)
        print('output of layer_norm.......', src_out)

        return src_out


class DecoderLayer(nn.Module):
    """Implement decoder sub-layers."""

    def __init__(self, model_dim, hidden_dim, nheads, p=0.1):
        super(DecoderLayer, self).__init__()
        self.mhatt_masked = MultiHeadAttention(
            model_dim, nheads, p, mask='triu',
        )
        self.mhatt = MultiHeadAttention(model_dim, nheads, p)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(p=p),
        )

        self.layer_norms = clones(nn.LayerNorm(model_dim), 3)

    def forward(self, tgt, enc):
        tgt_att1 = self.layer_norms[0](self.mhatt_masked(tgt, tgt, tgt) + tgt)
        tgt_att2 = self.layer_norms[1](self.mhatt(tgt_att1, enc, enc) + tgt_att1)
        tgt_out = self.layer_norms[2](self.ffn(tgt_att2) + tgt_att2)

        return tgt_out, enc


class Transformer(nn.Module):
    """Implement transformer model.

    Args:
        seq_len: number of unique tokens in vocabulary.
        model_dim: dimension of embedding.
        hidden_dim: size of hidden layer in feed forward sub-layers.
        nheads: number of attention heads.
        max_len: maximum sentence length used to pre-compute positional encoder.
        depth: number of encoder and decoder sub-layers.
    """

    def __init__(
            self,
            seq_len,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            p=0.1,
            max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.embedding = Embedding(seq_len, model_dim)
        self.pos_enc = PositionalEncoder(model_dim, max_len, p)

        self.encoder = seq_clones(
            EncoderLayer(model_dim, hidden_dim, nheads, p), depth,
        )
        self.decoder = seq_clones(
            DecoderLayer(model_dim, hidden_dim, nheads, p), depth,
        )

        self.apply(self._init_weights)

        self.src_embedding = None
        self.tgt_embedding = None

    def forward(self, src, tgt):
        right_shift = torch.zeros((tgt.shape[0], 1), dtype=torch.long, device=device)
        tgt_rs = torch.cat([right_shift, tgt], dim=1)[:, :-1]

        self.src_embedding = self.embedding(src)
        self.tgt_embedding = self.embedding(tgt_rs)

        src_pe = self.pos_enc(self.src_embedding)
        tgt_pe = self.pos_enc(self.tgt_embedding)

        dec, _ = self.decoder(tgt_pe, self.encoder(src_pe))

        return self.embedding(dec, inverse=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)


# only use the transformer encoder to encode the EEG signals
class TransformerEncoder(nn.Module):
    """Implement transformer model.

    Args:
        seq_len: number of unique tokens in vocabulary.
        model_dim: dimension of embedding.
        hidden_dim: size of hidden layer in feed forward sub-layers.
        nheads: number of attention heads.
        max_len: maximum sentence length used to pre-compute positional encoder.
        depth: number of encoder and decoder sub-layers.
    """

    def __init__(
            self,
            seq_len,
            model_dim,
            hidden_dim,
            nheads,
            depth,
            p=0.1,
            max_len=150,
            mask=None
    ):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(seq_len, model_dim)
        self.pos_enc = PositionalEncoder(model_dim, max_len, p)

        self.encoder = seq_clones(
            EncoderLayer(model_dim, hidden_dim, nheads, p, mask), depth,
        )

        self.apply(self._init_weights)

        self.src_embedding = None

    def forward(self, src):

        # self.src_embedding = self.embedding(src)
        print('src........', src)
        src_pe = self.pos_enc(src)
        print('src_pe.........', src_pe)
        enc = self.encoder(src_pe)

        return enc

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.1)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)