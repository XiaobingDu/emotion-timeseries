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

class EEGEncoder(nn.Module):
    """Transformer based model.
    left -- the channels in left brain region
    right -- the channels in right brain region
    embed_dim -- dimensions of input feature
    """

    def __init__(self, args, device=torch.cuda.set_device(0)): # torch.device('cuda:0')
        super(EEGEncoder, self).__init__()
        #the feature length of five brain regions
        self.time_steps = args['time_steps']  # 30
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

        self.time_transformer_enc = TransformerEncoder(self.time_steps, self.feature_dim, self.hidden_dim,  nheads=6, depth=2, p=0.5, max_len=self.feature_dim)
        self.time_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))

        self.channel_transformer_enc = TransformerEncoder(self.channels, self.feature_len, self.hidden_dim, nheads=6,depth=2, p=0.5, max_len=self.feature_len)
        self.channel_linear = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.feature_len, self.enc_dim, nn.LeakyReLU()))

        self.label_transformer = TransformerEncoder(self.labelNum, self.labelEmbedding, self.hidden_dim, nheads=6, depth=2, p=0.5,max_len=self.labelEmbedding, mask='co-label')
        self.label_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.labelEmbedding, self.enc_dim, nn.LeakyReLU()))

        self.concate_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.enc_dim * 2, self.enc_dim, nn.LeakyReLU()))

        # time_step = 30, channel = 30, 所以G[b_s, 30, 9]
        # 使用1DConv，需要对数据permute（0，2，1），得到 [b_s, 9, 30]
        self.convlayer = nn.Conv1d(in_channels=self.labelNum,out_channels = self.labelNum, kernel_size = 3, padding=1)



        # all_transformer --> out
        self.out = nn.Sequential(nn.Linear(256, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 64),
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

        # reshape the data to use the channel as the input sequence
        all_features = torch.reshape(all_features,[all_features.shape[0], all_features.shape[1], int(all_features.shape[2] / 5), 5])
        all_features = all_features.permute(0, 2, 1, 3)
        all_features = torch.reshape(all_features, [all_features.shape[0], all_features.shape[1],
                                                    all_features.shape[2] * all_features.shape[3]])
        # print('all_feature shape:', all_features.shape) # [64, 30, 150]
        channel_enc = self.channel_transformer_enc(all_features)
        # print('channel_enc shape:', channel_enc.shape) # [64, 30, 150]
        channel_enc = self.channel_linear(channel_enc)  # [64, 30, 256]

        concate_enc = torch.cat((time_enc, channel_enc), dim=-1)  # [64, 30, 512]
        concate_enc = self.concate_linear(concate_enc)  # [64, 30, 256]

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


        # self.G = torch.matmul(time_enc, label_enc)
        # self.G = torch.matmul(channel_enc, label_enc)
        self.G = torch.matmul(concate_enc, label_enc) # [64, 30, 9]
        self.G = self.G.permute(0, 2, 1)
        # learn the higher-order correlation matrix
        self.M = self.convlayer(self.G)
        self.M = self.M.permute(0, 2, 1) # [64, 30, 9]

        attn = torch.nn.functional.tanh(torch.nn.functional.softmax(self.M, dim=-1)) # [64, 30, 9]
        attn, _ = attn.max(2) # [64, 30]
        attn = torch.reshape(attn,[attn.shape[0],attn.shape[1],-1]) # [64, 30, 1]

        # print(attn.shape)
        # print('time_enc shape:', time_enc.shape) # [64, 30, 256]
        # context_feature = time_enc * attn # [64, 30, 256]
        context_feature = concate_enc * attn # [64, 30, 256]

        # print(context_feature.shape)
        predicted = self.out(context_feature).view(batch_size, seq_len, -1)
        # print('predicted shape:', predicted.shape) # [64, 30, 9]
        # predicted_last = predicted[:, -1, :]
        predicted_last = predicted.mean(1) # [64, 9]
        # print(predicted_last.shape)
        predict = predicted_last

        return predict
