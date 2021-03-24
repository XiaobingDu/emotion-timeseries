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

def regions(feature, dis, idx):
    channel = 30
    feature = feature
    dis = dis
    idx = idx
    Frontal = []
    Temporal = []
    Central = []
    Parietal = []
    Occipital = []

    # get the brain regions
    # 对应5个脑区的特征
    # idx = [[0,1,2,3,4,5,6],[7,11,12,16,17,21,22,26],[8,9,10,13,14,15,18,19,20],[23,24,25],[27,28,29]]
    F_idx = idx[0]  # [0,1,2,3,4,5,6]
    T_idx = idx[1]  # [7,11,12,16,17,21,22,26]
    C_idx = idx[2]  # [8,9,10,13,14,15,18,19,20]
    P_idx = idx[3]  # [23,24,25]
    O_idx = idx[4]  # [27,28,29]
    sample_nums = feature.shape[0]
    time_win = feature.shape[1]
    dim = feature.shape[2]
    PSD_dim = int(dim / channel)
    data = np.reshape(feature, [sample_nums, time_win, channel, PSD_dim])
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

    print('Frontal_feature shape:', Frontal_feature.shape)  # [68832, 10, 7, 5]
    Frontal_ch = Frontal_feature.shape[2]
    # reshape
    Frontal_feature = np.reshape(Frontal_feature, [Frontal_feature.shape[0], Frontal_feature.shape[1],
                                                   int(Frontal_feature.shape[2] * Frontal_feature.shape[3])])
    print('Frontal_feature shape:', Frontal_feature.shape)  # [68832, 10, 35]

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

    print('Temporal_feature shape:', Temporal_feature.shape)  # [68832, 10, 8, 5]
    Temporal_ch = Temporal_feature.shape[2]
    Temporal_feature = np.reshape(Temporal_feature, [Temporal_feature.shape[0], Temporal_feature.shape[1],
                                                     int(Temporal_feature.shape[2] * Temporal_feature.shape[3])])
    print('Temporal_feature shape:', Temporal_feature.shape)  # [68832, 10, 40]

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

    print('Central_feature shape:', Central_feature.shape)  # [68832, 10, 9, 5]
    Central_ch = Central_feature.shape[2]
    Central_feature = np.reshape(Central_feature, [Central_feature.shape[0], Central_feature.shape[1],
                                                   int(Central_feature.shape[2] * Central_feature.shape[3])])
    print('Central_feature shape:', Central_feature.shape)  # [68832, 10, 45]

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

    print('Parietal_feature shape:', Parietal_feature.shape)  # [68832, 10, 3, 5]
    Parietal_ch = Parietal_feature.shape[2]
    Parietal_feature = np.reshape(Parietal_feature, [Parietal_feature.shape[0], Parietal_feature.shape[1],
                                                     int(Parietal_feature.shape[2] * Parietal_feature.shape[3])])
    print('Parietal_feature shape:', Parietal_feature.shape)  # [68832, 10, 15]

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

    print('Occipital_feature shape:', Occipital_feature.shape)  # [68832, 10, 3, 5]
    Occipital_ch = Occipital_feature.shape[2]
    Occipital_feature = np.reshape(Occipital_feature, [Occipital_feature.shape[0], Occipital_feature.shape[1],
                                                       int(Occipital_feature.shape[2] * Occipital_feature.shape[3])])
    print('Occipital_feature shape:', Occipital_feature.shape)  # [68832, 10, 15]

    Frontal = Frontal_feature
    Temporal = Temporal_feature
    Central = Central_feature
    Parietal = Parietal_feature
    Occipital = Occipital_feature

    data.append(Frontal)
    data.append(Temporal)
    data.append(Central)
    data.append(Parietal)
    data.append(Occipital)
    data.append(dis)
    print('data len:', len(data))
    print('Frontal_feature shape:', data[0].shape)

    # combined = torch.cat((Frontal, Temporal, Central, Parietal, Occipital),dim=-1)
    # print('combined shape:', combined.shape) #([162509, 10, 150])

    return data