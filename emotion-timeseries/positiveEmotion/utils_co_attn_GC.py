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
def MediaEvalDataset(self, feature, dis, idx):

    print('*************')
    self.feature = feature
    self.dis = dis
    self.idx = idx
    self.num_samples = feature.shape[0]


    print('-----------')
    #对应5个脑区的特征
    #idx = [[0,1,2,3,4,5,6],[7,11,12,16,17,21,22,26],[8,9,10,13,14,15,18,19,20],[23,24,25],[27,28,29]]
    idx = self.idx

    F_idx = idx[0] #[0,1,2,3,4,5,6]
    T_idx = idx[1] #[7,11,12,16,17,21,22,26]
    C_idx = idx[2] #[8,9,10,13,14,15,18,19,20]
    P_idx = idx[3] #[23,24,25]
    O_idx = idx[4] #[27,28,29]
    sample_nums = self.feature.shape[0]
    time_win = self.feature.shape[1]
    dim = self.feature.sahpe[2]
    PSD_dim = dim/self.channel
    data = np.reshape(self.feature,[sample_nums,time_win,self.channel,PSD_dim])
    #array转换成torch tensor，为了使用 data.unsquezze_(2)和 torch.cat((),dim=2)
    data = torch.Tensor(data)
    # data = np.reshape(data,[self.channel, sample_nums*time_win*PSD_dim])

    #frontal channel
    for i in F_idx:
        print('F_idx:',i)
        if i == 0:
            Frontal_feature = data[:,:,i,:]
            Frontal_feature.unsqueeze_(2)
        else:
            Frontal_feature = torch.cat((Frontal_feature, data[:,:,i,:].unsquezze_(2)),dim=2)

    Frontal_ch = Frontal_feature.shape[2]
    #reshape
    # Frontal_feature = np.reshape(Frontal_feature,[sample_nums,time_win,Frontal_ch,PSD_dim])
    print('Frontal_feature shape:',Frontal_feature.shape)

    #Temporal channel
    for i in T_idx:
        print('T_idx:', i)
        if i == 0:
            Temporal_feature = data[:,:,i,:]
            Frontal_feature.unsqueeze_(2)
        else:
            Temporal_feature = torch.cat((Temporal_feature,data[:,:,i,:].unsquezze_(2)),dim=2)

    Temporal_ch = Temporal_feature.shape[2]
    print('Temporal_feature shape:', Temporal_feature.shape)

    # Central channel
    for i in C_idx:
        print('C_idx:', i)
        if i == 0:
            Central_feature = data[:, :, i, :]
            Central_feature.unsqueeze_(2)
        else:
            Central_feature = torch.cat((Central_feature, data[:, :, i, :].unsquezze_(2)), dim=2)

    Central_ch = Central_feature.shape[2]
    print('Central_feature shape:', Central_feature.shape)

    # Parietal channel
    for i in P_idx:
        print('P_idx:', i)
        if i == 0:
            Parietal_feature = data[:, :, i, :]
            Parietal_feature.unsqueeze_(2)
        else:
            Parietal_feature = torch.cat((Parietal_feature, data[:, :, i, :].unsquezze_(2)), dim=2)

    Parietal_ch = Parietal_feature.shape[2]
    print('Parietal_feature shape:', Parietal_feature.shape)

    # Occipital channel
    for i in O_idx:
        print('O_idx:', i)
        if i == 0:
            Occipital_feature = data[:, :, i, :]
            Occipital_feature.unsqueeze_(2)
        else:
            Occipital_feature = torch.cat((Occipital_feature, data[:, :, i, :].unsquezze_(2)), dim=2)

    Occipital_ch = Occipital_feature.shape[2]
    print('Occipital_feature shape:', Occipital_feature.shape)

    Frontal = Frontal_feature
    Temporal = Temporal_feature
    Central = Central_feature
    Parietal = Parietal_feature
    Occipital = Occipital_feature
    y = self.dis
    #将5个脑区的数据hstack
    combined = np.hstack([Frontal, Temporal, Central, Parietal, Occipital ])

    return combined, y, Frontal, Temporal, Central, Parietal, Occipital



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
    valid_loss_min_valence = checkpoint['valid_loss_min_valence']
    valid_loss_min_arousal = checkpoint['valid_loss_min_arousal']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min_valence.item(), valid_loss_min_arousal.item()

# if __name__ == '__mian__':
#     print('............')
#     from dataManager import dataSplit, get_sample_data
#     path1 = '/Volumes/DATA/EEG_Multi-label/EEG_LDL_9/EEG_PSD_multilabel_9_addLabel_sum1/'
#     path2 = '/Volumes/DATA/EEG_Multi-label/EEG_LDL_9/EEG_PSD_multilabel_9_win/featureAll.mat'
#     db_name = 'LDL_data'
#     idx = [[0, 1, 2, 3, 4, 5, 6], [7, 11, 12, 16, 17, 21, 22, 26], [8, 9, 10, 13, 14, 15, 18, 19, 20], [23, 24, 25],
#            [27, 28, 29]]
#
#     # load train, val, test data
#     data_set = get_sample_data(path1, path2)
#     train_data, test_data, train_label, test_label, train_dis, test_dis, train_score, test_score = dataSplit(path1,
#                                                                                                              data_set,db_name)
#     MediaEvalDataset(train_data,train_dis,idx)

