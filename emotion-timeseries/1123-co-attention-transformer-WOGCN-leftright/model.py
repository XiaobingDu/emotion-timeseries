#-*-coding:utf-8-*-

from __future__ import division
import torch
import torch.nn as nn
from modeling_transformer import TransformerEncoder
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

class EEGEncoder(nn.Module):
    """Transformer based model.
    left -- the channels in left brain region
    right -- the channels in right brain region
    embed_dim -- dimensions of input feature
    """

    def __init__(self, args, device=torch.cuda.set_device(0)): # torch.device('cuda:0')
        super(EEGEncoder, self).__init__()
        #the feature length of five brain regions
        self.left_len = args['left_len'] # 30
        self.right_len = args['right_len'] # 30
        self.feature_dim = args['feature_dim'] # 75 = 15channels *5 # 150 300 600
        self.out_layer = args['out_layer']
        self.sequence_len = args['sequence_len'] # 30 timesteps
        self.feature_len = args['feature_len'] # 150 = 30channels * 5
        self.enc_dim = args['enc_dim'] # 256
        self.hidden_dim = args['hidden_dim'] # 256
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        self.enc_all_linear1 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_len, self.enc_dim, nn.LeakyReLU()))
        # self.enc_all_linear2 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.enc_dim, 1, nn.LeakyReLU()))
        self.left_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))
        self.right_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))

        self.left_transformer_enc = TransformerEncoder(self.left_len, self.feature_dim, self.hidden_dim,  nheads=3, depth=2, p=0.5, max_len=75)
        self.right_transformer_enc = TransformerEncoder(self.right_len, self.feature_dim,self.hidden_dim, nheads=3, depth=2, p=0.5, max_len=75)
        self.all_transformer_enc = TransformerEncoder(self.sequence_len, self.feature_len,self.hidden_dim, nheads=3, depth=2, p=0.5, max_len=150)

        # [left right]-->att_linear
        # self.att_linear = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.enc_dim * 2, 1), nn.LeakyReLU())

        # lefr --> att_linear
        self.att_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.enc_dim, 1), nn.LeakyReLU())

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

    def forward(self, x, left_features, right_features, target=None, tgt_init=0.0):
        # Get batch dim
        x = x.float()
        left_features = left_features.float().cuda()
        right_features = right_features.float().cuda()
        # batch_size, seq_len
        batch_size, seq_len = x.shape[0], x.shape[1]

        # left transformer
        # print('left_feature shape:', left_features.shape)
        left_enc = self.left_transformer_enc(left_features)
        # print('*********left encoding:', left_enc)
        # print('left enc shape:', left_enc.shape)
        left_enc = self.left_linear(left_enc)
        # print('*********right encoding FC:', left_enc)
        # print('left enc shape:', left_enc.shape)

        # right transformer
        # print('right_feature shape:', right_features.shape)
        right_enc = self.right_transformer_enc(right_features)
        # print('*********right encoding:', right_enc)
        # print('right enc shape:', right_enc.shape)
        right_enc = self.right_linear(right_enc)
        # print('*********right encoding FC:', right_enc)
        # print('right enc shape:', right_enc.shape)

        # 尝试只使用left brain feature，因为left brain对于积极情绪的贡献更大
        concat_features = left_enc

        # concat left and right
        # Co-attention Scores
        # concat_features = torch.cat([left_enc, right_enc], dim=-1)

        att_score = self.att_linear(concat_features).squeeze(-1)
        # print('att_score....:', att_score.shape)
        att_score = torch.softmax(att_score, dim=-1)
        # print('att_score....:', att_score.shape)

        # when the left and right use the channel dim as the seq_len
        # all_transformer
        # left_features = torch.reshape(left_features,[left_features.shape[0],left_features.shape[1], int(left_features.shape[2]/5), 5])
        # left_features = left_features.permute(0,2,1,3)
        # left_features = torch.reshape(left_features,[left_features.shape[0],left_features.shape[1],left_features.shape[2]*left_features.shape[3]])
        # print('left_feature shape:', left_features.shape)
        # right_features = torch.reshape(right_features, [right_features.shape[0], right_features.shape[1], int(right_features.shape[2]/5), 5])
        # right_features = right_features.permute(0, 2, 1, 3)
        # right_features = torch.reshape(right_features, [right_features.shape[0], right_features.shape[1], right_features.shape[2]*right_features.shape[3]])
        # print('right_feature shape:', right_features.shape)

        # when the left and right use the time dim as the seq_len
        all_features = torch.cat([left_features, right_features], dim=-1)
        # print('all_feature shape:', all_features.shape)
        # reshape the data to use the channel as the input sequence
        all_features = torch.reshape(all_features, [all_features.shape[0],all_features.shape[1],int(all_features.shape[2]/5), 5])
        all_features = all_features.permute(0,2,1,3)
        all_features = torch.reshape(all_features,[all_features.shape[0], all_features.shape[1], all_features.shape[2]*all_features.shape[3]])
        # print('all_feature shape:', all_features.shape)
        presentation = self.all_transformer_enc(all_features)
        # print('presentation shape:', presentation.shape) # [32, 30, 150]
        # print('*********all present:', presentation)
        presentation = self.enc_all_linear1(presentation)
        # print('*********all present FC:', presentation)
        # print('presentation shape:', presentation.shape) # [32, 30, 1024]
        # presentation = self.enc_all_linear2(presentation)
        # print('presentation shape:', presentation.shape) # [32, 30, 1]
        presentation = torch.softmax(presentation, dim=-1)
        # print('*********all present softmax:', presentation)
        # print('presentation shape:', presentation.shape) # [32, 30, 1024]

        attn = att_score
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        # print('attention shape:', attn.shape) # [32, 30, 1]
        context = convolve(presentation, attn)
        # print('context shape:', context.shape)# [32, 30, 256]
        # print('********context:', context)
        predicted = self.out(context).view(batch_size, seq_len, -1)
        # print('predicted shape:', predicted.shape)
        predicted_last = predicted[:, -1, :]
        predict = predicted_last


        return predict, attn
