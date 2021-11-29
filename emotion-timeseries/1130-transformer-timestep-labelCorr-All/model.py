#-*-coding:utf-8-*-

from __future__ import division
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from modeling_transformer import TransformerEncoder
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

        ret, att = self.attention(
            self.linear_q(query).reshape(shape_q),
            self.linear_k(key).reshape(shape_k),
            self.linear_v(value).reshape(shape_v)
        )
        ret = ret.reshape(ret.shape[:2] + (self.model_dim,))

        return self.dropout(self.linear_out(ret)), att

    def attention(self, query, key, value):
        """Compute scaled dot-product attention.

        Args:
            query: tensor with shape (batch_size, sentence_len1, nheads, key_dim).
            key: tensor with shape (batch_size, sentence_len2, nheads, key_dim).
            value: tensor with shape (batch_size, sentence_len2, nheads, key_dim).

        Returns:
            tensor with shape (batch_size, sentence_len1, nheads, key_dim).
        """

        score = torch.einsum('bqhd,bkhd->bhqk', query, key)
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

        self.att = F.softmax(score / np.sqrt(score.shape[-1]), dim=-1)
        ret = torch.einsum('bhqk,bkhd->bqhd', self.att, value)

        return ret, self.att

class EEGEncoder(nn.Module):
    """Transformer based model.
    left -- the channels in left brain region
    right -- the channels in right brain region
    embed_dim -- dimensions of input feature
    """

    def __init__(self, args, device=torch.cuda.set_device(0)): # torch.device('cuda:0')
        super(EEGEncoder, self).__init__()
        #the feature length of five brain regions
        self.time_steps = args['time_steps'] # 30
        self.feature_dim = args['feature_dim'] # 150 = 30channels *5 # 150 300 600
        self.out_layer = args['out_layer']
        self.channels = args['channels'] # 30
        self.feature_len = args['feature_len'] # 150 = 30time_steps * 5
        self.labelNum = args['label_num'] # 9
        self.labelEmbedding = args['label_em'] # 300, Glove embedding
        self.enc_dim = args['enc_dim'] # 256
        self.hidden_dim = args['hidden_dim'] # 256
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        self.time_transformer_enc = TransformerEncoder(self.time_steps, self.feature_dim, self.hidden_dim,  nheads=3, depth=2, p=0.5, max_len=150)
        self.time_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))

        self.label_transformer = TransformerEncoder(self.labelNum, self.labelEmbedding, self.hidden_dim, nheads=3, depth=2, p=0.5,max_len=300, mask='co-label')
        self.label_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.labelEmbedding, self.enc_dim, nn.LeakyReLU()))

        # all_transformer --> out
        self.out = nn.Sequential(nn.Linear(256, 64),
                                 nn.LeakyReLU(),
                                 nn.Linear(64, 32),
                                 nn.LeakyReLU(),
                                 nn.Linear(32, self.out_layer)
                                 )

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, left_features, right_features, labelEmb=None):
        # Get batch dim
        x = x.float()
        left_features = left_features.float().cuda()
        right_features = right_features.float().cuda()
        labelEmb = labelEmb.float().cuda()
        # batch_size, seq_len
        batch_size, seq_len = x.shape[0], x.shape[1]

        # time_step as the input sequence
        all_features = torch.cat([left_features, right_features], dim=-1)
        # print('all_feature shape:', all_features.shape)  # [64, 30, 150]
        time_enc = self.time_transformer_enc(all_features)
        time_enc = self.time_linear(time_enc)
        # print('time_enc shape:', time_enc.shape) # [64, 30, 256]

        # print('******* label emb shape:', labelEmb.shape) # [9, 300]
        labelEmb_e = labelEmb.unsqueeze(dim=0)
        for i in range(batch_size):
            if i == 0:
                labelEmb = labelEmb_e
            else:
                labelEmb = torch.cat((labelEmb, labelEmb_e), dim=0)
        # print('******* label emb shape:', labelEmb.shape) # [64, 9, 300]
        label_corr = self.label_transformer(labelEmb)
        label_enc = self.label_linear(label_corr)
        label_enc = label_enc.permute(0, 2, 1)

        self.G = torch.matmul(time_enc, label_enc)
        # print('*******', self.G.shape) # [64, 30, 9]
        # print(self.G)
        attn = torch.nn.functional.tanh(torch.nn.functional.softmax(self.G, dim=-1)) # [64, 30, 9]
        attn, _ = attn.max(2) # [64, 30]
        attn = torch.reshape(attn,[attn.shape[0],attn.shape[1],-1]) # [64, 30, 1]
        # print(attn.shape)
        # print('time_enc shape:', time_enc.shape) # [64, 30, 256]
        context_feature = time_enc * attn # [64, 30, 256]
        # print(context_feature.shape)
        print(context_feature)
        predicted = self.out(context_feature).view(batch_size, seq_len, -1)
        # print('predicted shape:', predicted.shape) # [64, 30, 9]
        # predicted_last = predicted[:, -1, :]
        predicted_last = predicted.mean(1)
        print(predicted_last.shape)
        predict = predicted_last

        return predict
