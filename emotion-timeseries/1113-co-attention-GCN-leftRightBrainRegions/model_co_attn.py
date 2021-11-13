#-*-coding:utf-8-*-

from __future__ import division
import torch
import torch.nn as nn
from clstm import cLSTM
from graph_module import GCN
import warnings
warnings.filterwarnings('ignore')

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

class MovieNet(nn.Module):
    """Multimodal encoder-decoder LSTM model.
    modalities -- list of names of each input modality
    dims -- list of dimensions for input modalities
    embed_dim -- dimensions of embedding for feature-level fusion
    h_dim -- dimensions of LSTM hidden state
    n_layers -- number of LSTM layers
    attn_len -- length of local attention window
    """

    def __init__(self, args, device=torch.device('cuda:0')):
        super(MovieNet, self).__init__()
        #the feature length of five brain regions
        self.left_len = args['left_len']
        self.right_len = args['right_len']
        self.out_layer = args['out_layer']
        #concate the length
        self.total_mod_len = self.left_len + self.right_len

        #super parameters
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        #fully-connected linear layer
        self.left_linear = nn.Linear(self.left_len, self.h_dim, bias=True)
        self.right_linear = nn.Linear(self.right_len, self.h_dim, bias=True)

        #co-attention leayer
        self.att_linear1 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)

        #unimodal single-modality for cLSTM
        #unimodal vs. multi-modal
        self.unimodal_left = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_right = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())

        #Encoder Module
        #cLSTM module simultaneously
        self.shared_encoder = cLSTM(2,self.h_dim, batch_first=True).cuda(device=device)
        #shape = [1,h_dim]
        self.enc_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

        #Decoder Module
        self.decoder = nn.LSTM(2, self.h_dim, self.n_layers, batch_first=True)
        #decoder: ini parameters
        self.dec_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

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

        left_features=left_features.float()
        right_features=right_features.float()
        #batch_size , seq_len
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1) # 将enc_h0 在第一维上重复batch_size次，在第二维上重复1次
        c0 = self.enc_c0.repeat(1, batch_size, 1)

        # 1.linear transform: dim = h_dim
        left_features_rep = self.left_linear(left_features)
        right_features_rep = self.right_linear(right_features)

        #Co-attention Scores
        #eq.7
        #2. co-attention
        concat_features = torch.cat([left_features_rep, right_features_rep], dim=-1) # dim = -1; 第一维度拼接（横向拼接）；h_dim*2
        # concat_features = torch.tanh(concat_features)
        # att_1
        att_1 = self.att_linear1(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        #cLSTM Encoder
        #eq.5
        #befor input into cLSTM
        unimodal_left_input= left_features_rep
        unimodal_left_input = self.unimodal_left(unimodal_left_input).squeeze(-1)
        unimodal_left_input = torch.softmax(unimodal_left_input, dim=-1) #[32,20]

        unimodal_right_input= right_features_rep
        unimodal_right_input = self.unimodal_right(unimodal_right_input).squeeze(-1)
        unimodal_right_input = torch.softmax(unimodal_right_input, dim=-1) #[32,20]

        # 列向拼接
        #X1:T =x1:T ⊘x1:T ⊘···⊘x1:T
        # [32,100]
        enc_input_unimodal_cat=torch.cat([unimodal_left_input, unimodal_right_input], dim=-1)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, 2) #[32,20,5]
        #α=α1 ⊕α2 ⊕···⊕αm
        #att_1:[32, 20]; attn:[32,200]
        attn = att_1
        # [32, 20, 10]
        attn = attn.reshape(batch_size, seq_len, self.attn_len)

        #eq.6
        # output of cLSTM
        # [32, 10, 5]
        enc_out, _ = self.shared_encoder(enc_input_unimodal_cat)
        #eq.8
        #context vector d
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        # [32, 10, 5]
        context = convolve(enc_out, attn)
        # context_feature = context.reshape(-1, 5)  # [320,5]

        # Decoder
        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        #decoder
        # [32,10,5]
        dec_in = context.float()
        # [32, 10, 512]
        dec_out, _ = self.decoder(dec_in, (h0, c0)) #decoder
        # Undo the packing
        # [320,512]
        dec_out = dec_out.reshape(-1, self.h_dim) #[640,512]
        # eq.10
        ## [32,20,128]
        #without GCN
        # predicted = self.out(dec_out).view(batch_size, seq_len, self.out_layer)
        #with GCN
        predicted = self.out(dec_out).view(batch_size, seq_len, -1)
        ##[32,128]
        predicted_last = predicted[:, -1, :]
        #without GCN
        # predict = predicted_last

        # GCN module
        # num_class = 9
        GCN_module = GCN(num_classes=9, in_channel=300, t=0.4, adj_file='embedding/positiveEmotion_adj.pkl') #t-0.4
        GCN_output = GCN_module(inp='embedding/positiveEmotion_glove_word2vec.pkl')  # [9,256]
        GCN_output = self.GCN_out(GCN_output.cuda()) #[9,64]
        GCN_output = GCN_output.transpose(0, 1).cuda()  # [64,9]

        # GCN output * LSTM lastTimestep
        ## [32,9]
        predict = torch.matmul(predicted_last, GCN_output)  # ML-GCN eq.4


        return predict, enc_input_unimodal_cat, self.shared_encoder, att_1
