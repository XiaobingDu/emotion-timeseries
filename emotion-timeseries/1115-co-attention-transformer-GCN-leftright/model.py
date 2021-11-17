#-*-coding:utf-8-*-

from __future__ import division
import torch
import torch.nn as nn
from modeling_transformer import TransformerEncoder
from graph_module import GCN
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
        self.left_len = args['left_len'] # 15
        self.right_len = args['right_len'] # 15
        self.feature_dim = args['feature_dim'] #150
        self.out_layer = args['out_layer']
        self.sequence_len = args['sequence_len'] # 30 timesteps
        self.feature_len = args['feature_len'] # 30*5 = 150
        self.enc_dim = args['enc_dim']
        self.hidden_dim = args['hidden_dim']
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        self.enc_all_linear1 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_len, self.enc_dim, nn.LeakyReLU()))
        self.enc_all_linear2 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.enc_dim, 2, nn.LeakyReLU()))
        self.left_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))
        self.right_linear = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.feature_dim, self.enc_dim, nn.LeakyReLU()))

        self.left_transformer_enc = TransformerEncoder(self.left_len, self.feature_dim, self.hidden_dim,  nheads=5, depth=2, p=0.1, max_len=600)
        self.right_transformer_enc = TransformerEncoder(self.right_len, self.feature_dim,self.hidden_dim, nheads=5, depth=2, p=0.1, max_len=600)
        self.all_transformer_enc = TransformerEncoder(self.sequence_len, self.feature_len,self.hidden_dim, nheads=5, depth=2, p=0.1, max_len=600)

        # [left right]-->att_linear
        self.att_linear = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.hidden_dim * 2, 1), nn.LeakyReLU())

        # all_transformer --> out
        self.out = nn.Sequential(nn.Linear(1024, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, 64),
                                 # nn.LeakyReLU(),
                                 # nn.Linear(128, self.out_layer) #withoutGCN
                                 )
        self.GCN_out = nn.Sequential(nn.Linear(256, 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(128, 64))
                                  # nn.LeakyReLU(),
                                  # nn.Linear(256, 128))

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
        print('left_feature shape:', left_features.shape)
        left_enc = self.left_transformer_enc(left_features)
        print('left enc shape:', left_enc.shape)
        left_enc = self.left_linear(left_enc)
        print('left enc shape:', left_enc.shape)
        # right transformer
        print('right_feature shape:', right_features.shape)
        right_enc = self.right_transformer_enc(right_features)
        print('right enc shape:', right_enc.shape)
        right_enc = self.right_linear(right_enc)
        print('right enc shape:', right_enc.shape)

        # Co-attention Scores
        concat_features = torch.cat([left_enc, right_enc], dim=-1)
        att_score = self.att_linear(concat_features).squeeze(-1)
        att_score = torch.softmax(att_score, dim=-1)

        # all_transformer
        left_features = torch.reshape(left_features,[left_features.shape[0],left_features.shape[1], int(left_features.shape[2]/5), 5])
        left_features = left_features.permute(0,2,1,3)
        left_features = torch.reshape(left_features,[left_features.shape[0],left_features.shape[1],left_features.shape[2]*left_features.shape[3]])
        print('left_feature shape:', left_features.shape)
        right_features = torch.reshape(right_features, [right_features.shape[0], right_features.shape[1], int(right_features.shape[2]/5), 5])
        right_features = right_features.permute(0, 2, 1, 3)
        right_features = torch.reshape(right_features, [right_features.shape[0], right_features.shape[1], right_features.shape[2]*right_features.shape[3]])
        print('right_feature shape:', right_features.shape)
        all_features = torch.cat([left_features, right_features], dim=-1)
        print('all_feature shape:', all_features.shape)
        presentation = self.all_transformer_enc(all_features)
        print('presentation shape:', presentation.shape)
        presentation = self.enc_all_linear1(presentation)
        print('presentation shape:', presentation.shape)
        presentation = self.enc_all_linear2(presentation)
        print('presentation shape:', presentation.shape)
        presentation = torch.softmax(presentation)
        print('presentation shape:', presentation.shape)

        attn = att_score
        attn = attn.reshape(batch_size, seq_len, self.attn_len)
        print('attention shape:', attn.shape)
        context = convolve(presentation, attn)

        predicted = self.out(context).view(batch_size, seq_len, -1)
        predicted_last = predicted[:, -1, :]

        # GCN module
        # num_class = 9
        GCN_module = GCN(num_classes=9, in_channel=300, t=0.4, adj_file='embedding/positiveEmotion_adj.pkl') #t-0.4
        GCN_output = GCN_module(inp='embedding/positiveEmotion_glove_word2vec.pkl')  # [9,256]
        GCN_output = self.GCN_out(GCN_output.cuda()) #[9,64]
        GCN_output = GCN_output.transpose(0, 1).cuda()  # [64,9]

        # GCN output * LSTM lastTimestep
        ## [32,9]
        predict = torch.matmul(predicted_last, GCN_output)  # ML-GCN eq.4


        return predict, context, predict, attn
