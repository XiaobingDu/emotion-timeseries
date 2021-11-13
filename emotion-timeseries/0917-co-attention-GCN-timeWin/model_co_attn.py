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
        self.T1_len = args['T1_len']
        self.T2_len = args['T2_len']
        self.T3_len = args['T3_len']
        self.T4_len = args['T4_len']
        self.T5_len = args['T5_len']
        self.out_layer = args['out_layer']
        #concate the length
        self.total_mod_len = self.T1_len + self.T2_len +self.T3_len + self.T4_len + self.T5_len

        #super parameters
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        #fully-connected linear layer
        self.T1_linear = nn.Linear(self.T1_len, self.h_dim, bias=True)
        self.T2_linear = nn.Linear(self.T2_len, self.h_dim, bias=True)
        self.T3_linear = nn.Linear(self.T3_len, self.h_dim, bias=True)
        self.T4_linear = nn.Linear(self.T4_len, self.h_dim, bias=True)
        self.T5_linear = nn.Linear(self.T5_len, self.h_dim, bias=True)

        #co-attention leayer
        self.att_linear1 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear2 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear3 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear4 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear5 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear6 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim * 2, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim * 2, 1)
        self.att_linear7 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim * 2, 1),nn.LeakyReLU()) # nn.Linear(self.h_dim * 2, 1)
        self.att_linear8 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim * 2, 1),nn.LeakyReLU())  # nn.Linear(self.h_dim * 2, 1)
        self.att_linear9 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim * 2, 1),nn.LeakyReLU())  # nn.Linear(self.h_dim * 2, 1)
        self.att_linear10 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim * 2, 1),nn.LeakyReLU())  # nn.Linear(self.h_dim * 2, 1)

        #unimodal single-modality for cLSTM
        #unimodal vs. multi-modal
        self.unimodal_T1 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_T2 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_T3 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_T4 = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim,1)
        self.unimodal_T5 = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim, 1),nn.LeakyReLU()) # nn.Linear(self.h_dim,1)

        #Encoder Module
        #cLSTM module simultaneously
        self.shared_encoder = cLSTM(5,self.h_dim, batch_first=True).cuda(device=device)
        #shape = [1,h_dim]
        self.enc_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

        #Decoder Module
        self.decoder = nn.LSTM(5, self.h_dim, self.n_layers, batch_first=True)
        #decoder: ini parameters
        self.dec_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

        self.out = nn.Sequential(nn.Linear(1024, 512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(256, 128),
                                 nn.LeakyReLU(),
                                 nn.Linear(128, self.out_layer) #withoutGCN
                                 )
        self.GCN_out = nn.Sequential(nn.Linear(2048, 512),
                                  nn.LeakyReLU(),
                                  nn.Linear(512, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 128))

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, T1_feature, T2_feature, T3_feature, T4_feature, T5_feature, target=None, tgt_init=0.0):
        # Get batch dim
        x = x.float()

        T1_feature=T1_feature.float()
        T2_feature=T2_feature.float()
        T3_feature=T3_feature.float()
        T4_feature=T4_feature.float()
        T5_feature = T5_feature.float()
        #batch_size , seq_len
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1) # 将enc_h0 在第一维上重复batch_size次，在第二维上重复1次
        c0 = self.enc_c0.repeat(1, batch_size, 1)

        # 1.linear transform: dim = h_dim
        T1_feature_rep = self.T1_linear(T1_feature)
        T2_feature_rep = self.T2_linear(T2_feature)
        T3_feature_rep = self.T3_linear(T3_feature)
        T4_feature_rep = self.T4_linear(T4_feature)
        T5_feature_rep = self.T5_linear(T5_feature)

        #Co-attention Scores
        #eq.7
        #2. co-attention
        concat_features = torch.cat([T1_feature_rep, T2_feature_rep], dim=-1) # dim = -1; 第一维度拼接（横向拼接）；h_dim*2
        # concat_features = torch.tanh(concat_features)
        # att_1
        att_1 = self.att_linear1(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        concat_features = torch.cat([T1_feature_rep, T3_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_2 = self.att_linear2(concat_features).squeeze(-1)
        att_2 = torch.softmax(att_2, dim=-1)

        concat_features = torch.cat([T1_feature_rep, T4_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_3 = self.att_linear3(concat_features).squeeze(-1)
        att_3 = torch.softmax(att_3, dim=-1)

        concat_features = torch.cat([T1_feature_rep, T5_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_4 = self.att_linear4(concat_features).squeeze(-1)
        att_4 = torch.softmax(att_4, dim=-1)

        concat_features = torch.cat([T2_feature_rep, T3_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_5 = self.att_linear5(concat_features).squeeze(-1)
        att_5 = torch.softmax(att_5, dim=-1)

        concat_features = torch.cat([T2_feature_rep, T4_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_6 = self.att_linear6(concat_features).squeeze(-1)
        att_6 = torch.softmax(att_6, dim=-1)

        concat_features = torch.cat([T2_feature_rep, T5_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_7 = self.att_linear7(concat_features).squeeze(-1)
        att_7 = torch.softmax(att_7, dim=-1)

        concat_features = torch.cat([T3_feature_rep, T4_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_8 = self.att_linear8(concat_features).squeeze(-1)
        att_8 = torch.softmax(att_8, dim=-1)

        concat_features = torch.cat([T3_feature_rep, T5_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_9 = self.att_linear9(concat_features).squeeze(-1)
        att_9 = torch.softmax(att_9, dim=-1)

        concat_features = torch.cat([T4_feature_rep, T5_feature_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_10 = self.att_linear10(concat_features).squeeze(-1)
        att_10 = torch.softmax(att_10, dim=-1)

        #cLSTM Encoder
        #eq.5
        #befor input into cLSTM
        unimodal_T1_input= T1_feature_rep
        unimodal_T1_input = self.unimodal_T1(unimodal_T1_input).squeeze(-1)
        unimodal_T1_input = torch.softmax(unimodal_T1_input, dim=-1) #[32,20]

        unimodal_T2_input= T2_feature_rep
        unimodal_T2_input = self.unimodal_T2(unimodal_T2_input).squeeze(-1)
        unimodal_T2_input = torch.softmax(unimodal_T2_input, dim=-1) #[32,20]

        unimodal_T3_input= T3_feature_rep
        unimodal_T3_input = self.unimodal_T3(unimodal_T3_input).squeeze(-1)
        unimodal_T3_input = torch.softmax(unimodal_T3_input, dim=-1) #[32,20]

        unimodal_T4_input= T4_feature_rep
        unimodal_T4_input = self.unimodal_T4(unimodal_T4_input).squeeze(-1)
        unimodal_T4_input = torch.softmax(unimodal_T4_input, dim=-1) #[32,20]

        unimodal_T5_input = T5_feature_rep
        unimodal_T5_input = self.unimodal_T5(unimodal_T5_input).squeeze(-1)
        unimodal_T5_input = torch.softmax(unimodal_T5_input, dim=-1) #[32,20]

        # 列向拼接
        #X1:T =x1:T ⊘x1:T ⊘···⊘x1:T
        # [32,100]
        enc_input_unimodal_cat=torch.cat([unimodal_T1_input, unimodal_T2_input, unimodal_T3_input, unimodal_T4_input, unimodal_T5_input], dim=-1)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, 5) #[32,20,5]
        #α=α1 ⊕α2 ⊕···⊕αm
        #att_1:[32, 20]; attn:[32,200]
        attn=torch.cat([att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10], dim=-1)
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
        predicted = self.out(dec_out).view(batch_size, seq_len, self.out_layer)
        ##[32,128]
        predicted_last = predicted[:, -1, :]
        predict = predicted_last

        # GCN module
        # num_class = 9
        # GCN_module = GCN(num_classes=9, in_channel=300, t=0.4, adj_file='embedding/positiveEmotion_adj.pkl') #t-0.4
        # GCN_output = GCN_module(inp='embedding/positiveEmotion_glove_word2vec.pkl')  # [9,2048]
        # GCN_output = self.GCN_out(GCN_output.cuda()) #[9,128]
        # GCN_output = GCN_output.transpose(0, 1).cuda()  # [128,9]
        #
        # # GCN output * LSTM lastTimestep
        # ## [32,9]
        # predict = torch.matmul(predicted_last, GCN_output)  # ML-GCN eq.4


        return predict, enc_input_unimodal_cat, self.shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10