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
        self.Frontal_len = args['Frontal_len']
        self.Temporal_len = args['Temporal_len']
        self.Central_len = args['Central_len']
        self.Parietal_len = args['Parietal_len']
        self.Occipital_len = args['Occipital_len']
        self.out_layer = args['out_layer']
        #concate the length
        self.total_mod_len = self.Frontal_len + self.Temporal_len +self.Central_len + self.Parietal_len + self.Occipital_len

        #super parameters
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']
        self.dropout= args['dropout_prob']

        #fully-connected linear layer
        self.Frontal_linear = nn.Linear(self.Frontal_len, self.h_dim, bias=True)
        self.Temporal_linear = nn.Linear(self.Temporal_len, self.h_dim, bias=True)
        self.Central_linear = nn.Linear(self.Central_len, self.h_dim, bias=True)
        self.Parietal_linear = nn.Linear(self.Parietal_len, self.h_dim, bias=True)
        self.Occipital_linear = nn.Linear(self.Occipital_len, self.h_dim, bias=True)

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
        self.unimodal_Frontal = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_Temporal = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_Central = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU())
        self.unimodal_Parietal = nn.Sequential(nn.Dropout(self.dropout),nn.Linear(self.h_dim, 1), nn.LeakyReLU()) #nn.Linear(self.h_dim,1)
        self.unimodal_Occipital = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.h_dim, 1),nn.LeakyReLU()) # nn.Linear(self.h_dim,1)

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

        self.out = nn.Sequential(nn.Linear(5, 16),  # 128 #5 -> 1024 -> 16 same as the GCN hidden_size
                                 #  nn.LeakyReLU(),
                                 #  nn.Linear(512, 8),
                                 nn.LeakyReLU(),
                                 nn.Linear(16, self.out_layer))  # 1024 -> out_layer:2048

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, x, Frontal_features, Temporal_features, Central_features, Parietal_features, Occipital_features, target=None, tgt_init=0.0):
        # Get batch dim
        x = x.float()

        Frontal_features=Frontal_features.float()
        Temporal_features=Temporal_features.float()
        Central_features=Central_features.float()
        Parietal_features=Parietal_features.float()
        Occipital_features = Occipital_features.float()
        #batch_size , seq_len
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Set initial hidden and cell states for encoder
        h0 = self.enc_h0.repeat(1, batch_size, 1) # 将enc_h0 在第一维上重复batch_size次，在第二维上重复1次
        c0 = self.enc_c0.repeat(1, batch_size, 1)

        # 1.linear transform: dim = h_dim
        Frontal_features_rep = self.Frontal_linear(Frontal_features)
        Temporal_features_rep = self.Temporal_linear(Temporal_features)
        Central_features_rep = self.Central_linear(Central_features)
        Parietal_features_rep = self.Parietal_linear(Parietal_features)
        Occipital_features_rep = self.Occipital_linear(Occipital_features)

        #Co-attention Scores
        #eq.7
        #2. co-attention
        concat_features = torch.cat([Frontal_features_rep, Temporal_features_rep], dim=-1) # dim = -1; 第一维度拼接（横向拼接）；h_dim*2
        # concat_features = torch.tanh(concat_features)
        # att_1
        att_1 = self.att_linear1(concat_features).squeeze(-1)
        att_1 = torch.softmax(att_1, dim=-1)

        concat_features = torch.cat([Frontal_features_rep, Central_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_2 = self.att_linear2(concat_features).squeeze(-1)
        att_2 = torch.softmax(att_2, dim=-1)

        concat_features = torch.cat([Frontal_features_rep, Parietal_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_3 = self.att_linear3(concat_features).squeeze(-1)
        att_3 = torch.softmax(att_3, dim=-1)

        concat_features = torch.cat([Frontal_features_rep, Occipital_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_4 = self.att_linear4(concat_features).squeeze(-1)
        att_4 = torch.softmax(att_4, dim=-1)

        concat_features = torch.cat([Temporal_features_rep, Central_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_5 = self.att_linear5(concat_features).squeeze(-1)
        att_5 = torch.softmax(att_5, dim=-1)

        concat_features = torch.cat([Temporal_features_rep, Parietal_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_6 = self.att_linear6(concat_features).squeeze(-1)
        att_6 = torch.softmax(att_6, dim=-1)

        concat_features = torch.cat([Temporal_features_rep, Occipital_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_7 = self.att_linear7(concat_features).squeeze(-1)
        att_7 = torch.softmax(att_7, dim=-1)

        concat_features = torch.cat([Central_features_rep, Parietal_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_8 = self.att_linear8(concat_features).squeeze(-1)
        att_8 = torch.softmax(att_8, dim=-1)

        concat_features = torch.cat([Central_features_rep, Occipital_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_9 = self.att_linear9(concat_features).squeeze(-1)
        att_9 = torch.softmax(att_9, dim=-1)

        concat_features = torch.cat([Parietal_features_rep, Occipital_features_rep], dim=-1)
        # concat_features = torch.tanh(concat_features)
        att_10 = self.att_linear10(concat_features).squeeze(-1)
        att_10 = torch.softmax(att_10, dim=-1)

        #cLSTM Encoder
        #eq.5
        #befor input into cLSTM
        unimodal_Frontal_input= Frontal_features_rep
        unimodal_Frontal_input = self.unimodal_Frontal(unimodal_Frontal_input).squeeze(-1)
        unimodal_Frontal_input = torch.softmax(unimodal_Frontal_input, dim=-1) #[32,20]

        unimodal_Temporal_input= Temporal_features_rep
        unimodal_Temporal_input = self.unimodal_Temporal(unimodal_Temporal_input).squeeze(-1)
        unimodal_Temporal_input = torch.softmax(unimodal_Temporal_input, dim=-1) #[32,20]

        unimodal_Central_input= Central_features_rep
        unimodal_Central_input = self.unimodal_Central(unimodal_Central_input).squeeze(-1)
        unimodal_Central_input = torch.softmax(unimodal_Central_input, dim=-1) #[32,20]

        unimodal_Parietal_input= Parietal_features_rep
        unimodal_Parietal_input = self.unimodal_Parietal(unimodal_Parietal_input).squeeze(-1)
        unimodal_Parietal_input = torch.softmax(unimodal_Parietal_input, dim=-1) #[32,20]

        unimodal_Occipital_input = Occipital_features_rep
        unimodal_Occipital_input = self.unimodal_Occipital(unimodal_Occipital_input).squeeze(-1)
        unimodal_Occipital_input = torch.softmax(unimodal_Occipital_input, dim=-1) #[32,20]

        # 列向拼接
        #X1:T =x1:T ⊘x1:T ⊘···⊘x1:T
        # [32,100]
        enc_input_unimodal_cat=torch.cat([unimodal_Frontal_input, unimodal_Temporal_input, unimodal_Central_input, unimodal_Parietal_input, unimodal_Occipital_input], dim=-1)
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
        ## [32,10,5]
        predicted = context
        ##[32,5]
        predicted_last = predicted[:, -1, :]

        #GCN module
        #num_class = 9
        GCN_module = GCN(num_classes = 9, in_channel=300, t=0.4, adj_file='embedding/positiveEmotion_adj.pkl')
        GCN_output = GCN_module(inp='embedding/positiveEmotion_glove_word2vec.pkl') #[9,5]
        GCN_output = GCN_output.transpose(0, 1).cuda() #[5,9]

        # GCN output * LSTM out lastTimestep
        ## [32,9]
        predict = torch.matmul(predicted_last, GCN_output)  # ML-GCN eq.4
        # softmax layer
        softmax = torch.nn.Softmax(dim=1)
        predicted = softmax(predict)

        # #log_softmax layer
        # log_softmax = torch.nn.LogSoftmax(dim=1)
        # predicted = log_softmax(predict)

        return predict, enc_input_unimodal_cat, self.shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10
