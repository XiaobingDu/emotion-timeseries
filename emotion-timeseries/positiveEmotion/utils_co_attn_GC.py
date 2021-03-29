#-*-coding:utf-8-*-

from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
from keras.utils import to_categorical
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
import shutil

#___________________________________________________________________________________________________________________________

# New Dataloader for MovieClass
class MediaEvalDataset(Dataset):
    def __init__(self,feature, dis, idx):
        self.channel = 30
        self.feature = feature
        self.dis = dis
        self.idx = idx
        self.Frontal = []
        self.Temporal = []
        self.Central = []
        self.Parietal = []
        self.Occipital = []

        #get the brain regions
        # 对应5个脑区的特征
        # idx = [[0,1,2,3,4,5,6],[7,11,12,16,17,21,22,26],[8,9,10,13,14,15,18,19,20],[23,24,25],[27,28,29]]
        F_idx = idx[0]  # [0,1,2,3,4,5,6]
        T_idx = idx[1]  # [7,11,12,16,17,21,22,26]
        C_idx = idx[2]  # [8,9,10,13,14,15,18,19,20]
        P_idx = idx[3]  # [23,24,25]
        O_idx = idx[4]  # [27,28,29]
        sample_nums = self.feature.shape[0]
        time_win = self.feature.shape[1]
        dim = self.feature.shape[2]
        PSD_dim = int(dim / self.channel)
        data = np.reshape(self.feature, [sample_nums, time_win, self.channel, PSD_dim])
        # array转换成torch tensor，为了使用 data.unsquezze_(2)和 torch.cat((),dim=2)
        data = torch.Tensor(data)

        cnt = 0
        # frontal channel
        for f in F_idx:
            print('F_idx:', f)
            if cnt == 0:
                Frontal_feature = data[:, :, f, :]
                Frontal_feature.unsqueeze_(2)
                cnt = 1
            else:
                Frontal_feature = torch.cat((Frontal_feature, data[:, :, f, :].unsqueeze_(2)), dim=2)

        print('Frontal_feature shape:', Frontal_feature.shape) #[68832, 10, 7, 5]
        Frontal_ch = Frontal_feature.shape[2]
        # reshape
        Frontal_feature = np.reshape(Frontal_feature, [Frontal_feature.shape[0], Frontal_feature.shape[1],
                                                         int(Frontal_feature.shape[2] * Frontal_feature.shape[3])])
        print('Frontal_feature shape:', Frontal_feature.shape) #[68832, 10, 35]

        cnt = 0
        # Temporal channel
        for t in T_idx:
            print('T_idx:', t)
            if cnt == 0:
                Temporal_feature = data[:, :, t, :]
                Temporal_feature.unsqueeze_(2)
                cnt = 1
            else:
                Temporal_feature = torch.cat((Temporal_feature, data[:, :, t, :].unsqueeze_(2)), dim=2)

        print('Temporal_feature shape:', Temporal_feature.shape) #[68832, 10, 8, 5]
        Temporal_ch = Temporal_feature.shape[2]
        Temporal_feature = np.reshape(Temporal_feature, [Temporal_feature.shape[0], Temporal_feature.shape[1],
                                                       int(Temporal_feature.shape[2] * Temporal_feature.shape[3])])
        print('Temporal_feature shape:', Temporal_feature.shape) #[68832, 10, 40]

        cnt = 0
        # Central channel
        for c in C_idx:
            print('C_idx:', c)
            if cnt == 0:
                Central_feature = data[:, :, c, :]
                Central_feature.unsqueeze_(2)
                cnt = 1
            else:
                Central_feature = torch.cat((Central_feature, data[:, :, c, :].unsqueeze_(2)), dim=2)

        print('Central_feature shape:', Central_feature.shape) #[68832, 10, 9, 5]
        Central_ch = Central_feature.shape[2]
        Central_feature = np.reshape(Central_feature, [Central_feature.shape[0], Central_feature.shape[1],
                                                         int(Central_feature.shape[2] * Central_feature.shape[3])])
        print('Central_feature shape:', Central_feature.shape) #[68832, 10, 45]

        cnt = 0
        # Parietal channel
        for p in P_idx:
            print('P_idx:', p)
            if cnt == 0:
                Parietal_feature = data[:, :, p, :]
                Parietal_feature.unsqueeze_(2)
                cnt = 1
            else:
                Parietal_feature = torch.cat((Parietal_feature, data[:, :, p, :].unsqueeze_(2)), dim=2)

        print('Parietal_feature shape:', Parietal_feature.shape) #[68832, 10, 3, 5]
        Parietal_ch = Parietal_feature.shape[2]
        Parietal_feature = np.reshape(Parietal_feature,[Parietal_feature.shape[0],Parietal_feature.shape[1],int(Parietal_feature.shape[2]*Parietal_feature.shape[3])])
        print('Parietal_feature shape:', Parietal_feature.shape) #[68832, 10, 15]

        cnt = 0
        # Occipital channel
        for o in O_idx:
            print('O_idx:', o)
            if cnt == 0:
                Occipital_feature = data[:, :, o, :]
                Occipital_feature.unsqueeze_(2)
                cnt = 1
            else:
                Occipital_feature = torch.cat((Occipital_feature, data[:, :, o, :].unsqueeze_(2)), dim=2)

        print('Occipital_feature shape:', Occipital_feature.shape) #[68832, 10, 3, 5]
        Occipital_ch = Occipital_feature.shape[2]
        Occipital_feature = np.reshape(Occipital_feature,[Occipital_feature.shape[0],Occipital_feature.shape[1],int(Occipital_feature.shape[2]*Occipital_feature.shape[3])])
        print('Occipital_feature shape:', Occipital_feature.shape) #[68832, 10, 15]

        self.Frontal = Frontal_feature
        self.Temporal = Temporal_feature
        self.Central = Central_feature
        self.Parietal = Parietal_feature
        self. Occipital = Occipital_feature



    def __len__(self):
        num_samples = self.feature.shape[0]
        return num_samples

    def __getitem__(self, index):
        F = self.Frontal[index]
        T = self.Temporal[index]
        C = self.Central[index]
        P = self.Parietal[index]
        O = self.Occipital[index]
        y = self.dis[index]

        # 将5个脑区的数据hstack
        # combined = np.hstack([F, T, C, P, O])
        combined = torch.cat((F, T, C, P, O), dim=-1) #[10,150]

        return combined, y, F, T, C, P, O



#________________________________________________________________________________________________________________________________________


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch == 100:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

#calculate CCC for SEND dataset
def prsn(emot_score, labels):
    """Computes concordance correlation coefficient."""

    labels_mu = torch.mean(labels)
    emot_mu = torch.mean(emot_score)
    vx = emot_score - emot_mu
    vy = labels - labels_mu
    # prsn_corr = torch.mean(vx * vy)
    prsn_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return prsn_corr

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min_kl = checkpoint['valid_loss_min_kl']

    return model, optimizer, checkpoint['epoch'], valid_loss_min_kl.item()


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'), encoding='bytes')
    print(result)
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj