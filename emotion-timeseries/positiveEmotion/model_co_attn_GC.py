#-*-coding:utf-8-*-


from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse


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
        #对应5个脑区的脑电特征长度
        self.Frontal_len = args['Frontal_len']
        self.Temporal_len = args['Temporal_len']
        self.Central_len = args['Central_len']
        self.Parietal_len = args['Parietal_len']
        self.Occipital_len = args['Occipital_len']
        self.out_layer = args['out_layer']
        #将5个脑区的特征长度相加
        self.total_mod_len = self.Frontal_len + self.Temporal_len +self.Central_len + self.Parietal_len + self.Occipital_len
        # self.total_mod_len = 3*(self.text_len)

        #super parameters
        self.embed_dim = args['embed_dim']
        self.h_dim = args['h_dim']
        self.n_layers = args['n_layers']
        self.attn_len = args['attn_len']
        self.dropout=0.5

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
        # 5 代表使用5个脑区的特征
        self.shared_encoder = cLSTM(5,self.h_dim, batch_first=True).cuda(device=device)
        #To ensure update parameters during training: nn.Parameter
        #encoder: init parameters
        #shape = [1,h_dim]
        self.enc_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.enc_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

        #Decoder Module
        # Decodes targets and LSTM hidden states
        # self.decoder = nn.LSTM(1 + self.h_dim, self.h_dim, self.n_layers, batch_first=True)
        #6 represents the context vector length
        #6 代表的是 input_size，x的特征维度
        self.decoder = nn.LSTM(6, self.h_dim, self.n_layers, batch_first=True)
        #decoder: ini parameters
        self.dec_h0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))
        self.dec_c0 = nn.Parameter(torch.rand(self.n_layers, 1, self.h_dim))

        # Final MLP output network
        self.out = nn.Sequential(nn.Linear(self.h_dim, 2),#128 #h_dim -> 2
                                #  nn.LeakyReLU(),
                                #  nn.Linear(512, 8),
                                 nn.LeakyReLU(),
                                 nn.Linear(2, self.out_layer)) #2 -> out_layer

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
        # Convert raw features into equal-dimensional embeddings
        # embed = self.embed(x)

        # 1.linear transform: dim = h_dim
        Frontal_features_rep = self.face_linear(Frontal_features)
        Temporal_features_rep = self.va_linear(Temporal_features)
        Central_features_rep = self.audio_linear(Central_features)
        Parietal_features_rep = self.scene_linear(Parietal_features)
        Occipital_features_rep = self.scene_linear(Occipital_features)

        #Co-attention Scores
        #eq.7
        #2. co-attention
        concat_features = torch.cat([Frontal_features_rep, Temporal_features_rep], dim=-1) # dim = -1; 第一维度拼接；h_dim*2
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
        #befor  input into cLSTM
        unimodal_Frontal_input= Frontal_features_rep
        unimodal_Frontal_input = self.unimodal_va(unimodal_Frontal_input).squeeze(-1)
        unimodal_Frontal_input = torch.softmax(unimodal_Frontal_input, dim=-1)

        unimodal_Temporal_input= Temporal_features_rep
        unimodal_Temporal_input = self.unimodal_face(unimodal_Temporal_input).squeeze(-1)
        unimodal_Temporal_input = torch.softmax(unimodal_Temporal_input, dim=-1)

        unimodal_Central_input= Central_features_rep
        unimodal_Central_input = self.unimodal_audio(unimodal_Central_input).squeeze(-1)
        unimodal_Central_input = torch.softmax(unimodal_Central_input, dim=-1)

        unimodal_Parietal_input= Parietal_features_rep
        unimodal_Parietal_input = self.unimodal_scene(unimodal_Parietal_input).squeeze(-1)
        unimodal_Parietal_input = torch.softmax(unimodal_Parietal_input, dim=-1)

        unimodal_Occipital_input = Occipital_features_rep
        unimodal_Occipital_input = self.unimodal_scene(unimodal_Occipital_input).squeeze(-1)
        unimodal_Occipital_input = torch.softmax(unimodal_Occipital_input, dim=-1)
        # print(unimodal_face_input.shape, unimodal_va_input.shape, unimodal_audio_input.shape, unimodal_scene_input.shape)

        # 列向拼接
        #X1:T =x1:T ⊘x1:T ⊘···⊘x1:T
        enc_input_unimodal_cat=torch.cat([unimodal_Frontal_input, unimodal_Temporal_input, unimodal_Central_input, unimodal_Parietal_input, unimodal_Occipital_input], dim=-1)
        # print(enc_input_unimodal_cat.shape, batch_size, seq_len)
        enc_input_unimodal_cat = enc_input_unimodal_cat.reshape(batch_size, seq_len, 5)
        #α=α1 ⊕α2 ⊕···⊕αm
        attn=torch.cat([att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10], dim=-1)
        attn = attn.reshape(batch_size, seq_len, self.attn_len)

        # cLSTM Encoder
        #eq.6
        # output of cLSTM
        enc_out, _ = self.shared_encoder(enc_input_unimodal_cat)
        # Undo the packing
        # enc_out, _ = pad_packed_sequence(enc_out, batch_first=True)

        #eq.8
        #context vector d
        # Convolve output with attention weights
        # i.e. out[t] = a[t,0]*in[t] + ... + a[t,win_len-1]*in[t-(win_len-1)]
        context = convolve(enc_out, attn)

        #Decoder
        # Set initial hidden and cell states for decoder
        h0 = self.dec_h0.repeat(1, batch_size, 1)
        c0 = self.dec_c0.repeat(1, batch_size, 1)
        if target is not None:
            # print(target[0].shape)
            # exit()
            #target == GT labels
            target_0 = target[0].float().reshape(batch_size, seq_len, 1)
            target_0 = torch.nn.Parameter(target_0).cuda()
            target_1 = target[1].float().reshape(batch_size, seq_len, 1)
            target_1 = torch.nn.Parameter(target_1).cuda()
            # print(pad_shift(target, 1, tgt_init), context.shape)

            #eq.9
            # Concatenate targets from previous timesteps to context
            #将previous time label与context拼接
            #targets from previous timesteps 怎样得到的呢？怎样传入呢？
            dec_in = torch.cat([pad_shift(target_0, 1, tgt_init),pad_shift(target_1, 1, tgt_init), context], 2)
            dec_out, _ = self.decoder(dec_in, (h0, c0))
            # Undo the packing
            dec_out = dec_out.reshape(-1, self.h_dim)

            # dec_in = context           
            # dec_out, _ = self.decoder(dec_in, (h0, c0))            
            # dec_out = dec_out.reshape(-1, self.h_dim)
            #eq.10
            #
            predicted = self.out(dec_out).view(batch_size, seq_len, self.out_layer)
        else:
            # Use earlier predictions to predict next time-steps
            predicted = []
            p = torch.ones(batch_size, 1).to(self.device) * tgt_init
            h, c = h0, c0
            for t in range(seq_len):
                # Concatenate prediction from previous timestep to context
                i = torch.cat([p, context[:, t, :]], dim=1).unsqueeze(1)
                # Get next decoder LSTM state and output
                o, (h, c) = self.decoder(i, (h, c))
                # Computer prediction from output state
                p = self.out(o.view(-1, self.h_dim))
                predicted.append(p.unsqueeze(1))
            predicted = torch.cat(predicted, dim=1) ##save the predict of every timesteps
        # Mask target entries that exceed sequence lengths
        # predicted = predicted * mask.float()
        return predicted, enc_input_unimodal_cat, self.shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10
