
# -*-coding:utf-8-*-


from __future__ import print_function
import torch
from model_co_attn_GC import MovieNet
from dataManager import dataSplit, get_sample_data
from utils_co_attn_GC import adjust_learning_rate, MediaEvalDataset, prsn, save_ckp, load_ckp
from torch.utils.data import DataLoader
import time
import math
import warnings
import pickle
import numpy as np
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
from scipy.stats.stats import pearsonr
# from block import fusions
import torch.nn.functional as F
from scipy.stats.mstats import pearsonr
from clstm import cLSTM, train_model_gista, train_model_adam, cLSTMSparse

args = {}

path1 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_addLabel_sum1/'
path2 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/featureAll.mat'
db_name = 'LDL_data'
best_model_path ="./best_model"
checkpoint_path ="./checkpoints"
valid_loss_min =np.Inf

## Network Arguments
args['Frontal_len'] = 35
args['Temporal_len'] = 40
args['Central_len'] = 45
args['Parietal_len'] = 15
args['Occipital_len'] = 15
args['out_layer'] = 9
args['dropout_prob'] = 0.5
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['train_flag'] = True
args['model_path'] = 'trained_models/EEG_eval_model.tar'
args['optimizer'] = 'adam'
args['embed_dim'] = 512
args['h_dim'] = 512
args['n_layers'] = 1
args['attn_len'] = 10
num_epochs = 20
batch_size = 32
lr =1e-4
GC_est =None
# 对应5个脑区的电极idx：Frontal、Temporal、Central、Parietal、Occipital
idx = [[0 ,1 ,2 ,3 ,4 ,5 ,6] ,[7 ,11 ,12 ,16 ,17 ,21 ,22 ,26] ,[8 ,9 ,10 ,13 ,14 ,15 ,18 ,19 ,20] ,[23 ,24 ,25]
       ,[27 ,28 ,29]]

# load train, val, test data
data_set = get_sample_data(path1 ,path2)
train_data, val_data, test_data, train_dis, val_dis, test_dis = dataSplit(path1 ,data_set ,db_name)

#通过 MediaEvalDataset 将数据进行加载，返回Dataset对象，包含data和labels
trSet = MediaEvalDataset(train_data, train_dis, idx)
valSet = MediaEvalDataset(val_data, val_dis, idx)
testSet = MediaEvalDataset(test_data, test_dis, idx)

#读取数据
trDataloader = DataLoader(trSet ,batch_size=batch_size ,shuffle=True ,num_workers=2) # len = 5079 (bath)
valDataloader = DataLoader(valSet ,batch_size=batch_size ,shuffle=True ,num_workers=2) #len = 3172
testDataloader = DataLoader(testSet ,batch_size=batch_size ,shuffle=True ,num_workers=2) #len = 2151

# Initialize network
net = MovieNet(args)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
optimizer = torch.optim.RMSprop(net.parameters(), lr=lr) if args['optimizer' ]== 'rmsprop' else torch.optim.Adam \
    (net.parameters() ,lr=lr, weight_decay=0.9)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
# crossEnt = torch.nn.BCELoss()
# mse = torch.nn.MSELoss(reduction='sum')
kl_div = torch.nn.KLDivLoss(size_average = True, reduce = True)


for epoch_num in range(num_epochs):
    #    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_learning_rate(optimizer, epoch_num, lr)

    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train()  # the state of model
    # Variables to track training performance:
    avg_tr_loss = 0
    for i, data in enumerate(trDataloader):
        print("第 {} 个Batch.....".format(i))
        st_time = time.time()
        train, dis, Frontal, Temporal, Central, Parietal, Occipital = data  # get training date
        # labels1 = labels[0]
        # labels2= labels[1]
        if args['use_cuda']: # use cuda
            train = torch.nn.Parameter(train).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            # labels1 = torch.nn.Parameter(labels1).cuda()
            # labels2 = torch.nn.Parameter(labels2).cuda()
            Frontal = torch.nn.Parameter(Frontal).cuda()
            Temporal = torch.nn.Parameter(Temporal).cuda()
            Central = torch.nn.Parameter(Central).cuda()
            Parietal = torch.nn.Parameter(Parietal).cuda()
            Occipital = torch.nn.Parameter(Occipital).cuda()

        train.requires_grad_()  # backward
        dis.requires_grad_()
        # labels1.requires_grad_()
        # labels2.requires_grad_()
        Frontal.requires_grad_()
        Temporal.requires_grad_()
        Central.requires_grad_()
        Parietal.requires_grad_()
        Occipital.requires_grad_()


        # Forward pass
        print('train dis........',dis.shape)
        emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 \
            = net(train, Frontal, Temporal, Central, Parietal, Occipital, dis)
        train_model_gista(shared_encoder, input_clstm, lam=0.5, lam_ridge=1e-4, lr=0.001, max_iter=1, check_every=1000, truncation=64)
        GC_est = shared_encoder.GC().cpu().data.numpy()

        emot_dis = emot_dis.squeeze(dim=0)
        print('emot_dis shape.....', emot_dis.shape)
        # labels1 = labels1.T
        # labels2 = labels2.T
        dis = np.reshape(dis,[dis.shape[0],dis.shape[2]])
        print('dis shape.....', dis.shape)

        # mamx-min norm
        # emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
        # mse loss
        # 两种label的loss之和
        # l = mse(emot_score[:,0].unsqueeze(dim=1), labels1) + mse(emot_score[:,1].unsqueeze(dim=1), labels2)
        # kldiv loss
        l = kl_div(emot_dis, dis)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        # scheduler.step()
        avg_tr_loss += l.item()

    # print(GC_est)
    print("Epoch no:" ,epoch_num +1, "| Avg train loss:" ,format(avg_tr_loss /len(trSet) ,'0.4f') )

    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.eval()
    # valmse = 0
    # aromse = 0
    val_kl = 0
    # valpcc = 0
    # aropcc = 0
    emopcc = 0
    for i, data in enumerate(valDataloader):
        st_time = time.time()
        # val, labels,  F, Va, scene, audio  = data
        val, dis, Frontal, Temporal, Central, Parietal, Occipital = data
        # labels1 = labels[0]
        # labels2 = labels[1]
        if args['use_cuda']:
            val = torch.nn.Parameter(val).cuda()
            # labels1 = torch.nn.Parameter(labels1).cuda()
            # labels2 = torch.nn.Parameter(labels2).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            Frontal = torch.nn.Parameter(Frontal).cuda()
            Temporal = torch.nn.Parameter(Temporal).cuda()
            Central = torch.nn.Parameter(Central).cuda()
            Parietal = torch.nn.Parameter(Parietal).cuda()
            Occipital = torch.nn.Parameter(Occipital).cuda()

        # Forward pass
        emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
            net(val, Frontal, Temporal, Central, Parietal, Occipital, dis)

        emot_dis = emot_dis.squeeze(dim=0)
        # labels1 = labels1.T
        # labels2 = labels2.T
        dis = dis.T

        # emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
        # labels1 = (2*(labels1 - torch.min(labels1)))/(torch.max(labels1) - torch.min(labels1)) -1
        # labels2 = (2*(labels2 - torch.min(labels2)))/(torch.max(labels2) - torch.min(labels2)) -1
        # 每一个batch的average mse loss相加
        # valmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
        # aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]
        val_kl = kl_div(emot_dis, dis) /dis.shape[0]

        # Pearson correlation
        # valpcc += pearsonr(emot_score[:, 0].unsqueeze(dim=1).cpu().detach().numpy(), labels1.cpu().detach().numpy())[0]
        # aropcc += pearsonr(emot_score[:, 1].unsqueeze(dim=1).cpu().detach().numpy(), labels2.cpu().detach().numpy())[0]
        emopcc += pearsonr(emot_dis.unsqueeze(dim=1).cpu().detach().numpy(), dis.cpu().detach().numpy())[0]

    # 每一个epoch loss平均
    # 每一个epoch pcc平均
    # epoch_valmse = valmse/len(valSet)
    # epoch_aromse = aromse/len(valSet)
    epoch_kl = val_kl /len(valSet)
    # epoch_valpcc = valpcc / len(valSet)
    # epoch_aropcc = aropcc / len(valSet)
    epoch_pcc = emopcc / len(valSet)
    # validation loss
    val_loss =epoch_kl

    # val_loss=(epoch_aromse+epoch_valmse)/2
    print("Epoch emotion distribution KLDivLoss:", epoch_kl.item() , "\nEpoch emotion distribution PCC:", epoch_pcc.item() ,"\n", "==========================")

    # checkpoint
    checkpoint = {
        'epoch': epoch_num + 1,
        'valid_loss_min_kl': epoch_kl,
        # 'valid_loss_min_arousal': epoch_aromse,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path + "/train_co_attn_GC_current_checkpoint.pt",
             best_model_path + "/train_co_attn_GC_best_model.pt")

    ## TODO: save the model if validation loss has decreased
    # 比较目前val_loss 与 valid_loss_min
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, val_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path + "/train_co_attn_GC_current_checkpoint.pt",
                 best_model_path + "/train_co_attn_GC_best_model.pt")
        valid_loss_min = val_loss

net = MovieNet(args)
net, optimizer, start_epoch, valid_loss_min_valence, valid_loss_min_arousal = load_ckp(
    best_model_path + "/train_co_attn_GC_best_model.pt", net, optimizer)
# 至此，training 和 validation结束

# testing
net.eval()
# testmse = 0
# aromse = 0
test_kl = 0
# testpcc = 0
# aropcc = 0
emopcc = 0
for i, data in enumerate(testDataloader):
    st_time = time.time()
    test, dis, Frontal, Temporal, Central, Parietal, Occipital = data
    # labels1 = labels[0]
    # labels2 = labels[1]
    dis = dis
    if args['use_cuda']:
        test = torch.nn.Parameter(test).cuda()
        # labels1 = torch.nn.Parameter(labels1).cuda()
        # labels2 = torch.nn.Parameter(labels2).cuda()
        dis = torch.nn.Parameter(dis).cuda()
        Frontal = torch.nn.Parameter(Frontal).cuda()
        Temporal = torch.nn.Parameter(Temporal).cuda()
        Central = torch.nn.Parameter(Central).cuda()
        Parietal = torch.nn.Parameter(Parietal).cuda()
        Occipital = torch.nn.Parameter(Occipital).cuda()

    # Forward pass
    emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
        net(test, Frontal, Temporal, Central, Parietal, Occipital, dis)
    print(att_1, att_2, att_3, att_4, att_5, att_6)

    emot_dis = emot_dis.squeeze(dim=0)
    # labels1 = labels1.T
    # labels2 = labels2.T
    dis = dis.T

    # min-max norm
    # emot_score = (2*(emot_score - torch.min(emot_score))/(torch.max(emot_score) - torch.min(emot_score))) -1
    # print(emot_score)
    # labels1 = (2*(labels1 - torch.min(labels1)))/(torch.max(labels1) - torch.min(labels1)) -1
    # labels2 = (2*(labels2 - torch.min(labels2)))/(torch.max(labels2) - torch.min(labels2)) -1
    # mse loss
    # testmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
    # aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]
    # kldiv loss
    test_kl = kl_div(emot_dis, dis) / dis.shape[0]

    # pearson correlation
    # testpcc += pearsonr(emot_score[:, 0].unsqueeze(dim=1).cpu().detach().numpy(), labels1.cpu().detach().numpy())[0]
    # aropcc += pearsonr(emot_score[:, 1].unsqueeze(dim=1).cpu().detach().numpy(), labels2.cpu().detach().numpy())[0]
    emopcc += pearsonr(emot_dis.unsqueeze(dim=1).cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
# average loss
# test_testmse = testmse/len(testSet)
# test_aromse = aromse/len(testSet)
test_testkl = test_kl / len(testSet)
# average pcc
# test_testpcc = testpcc / len(testSet)
# test_aropcc = aropcc / len(testSet)
test_emopcc = emopcc / len(testSet)

print("Test Emotion distribution KLDivLoss:", test_testkl.item(), "\Test Emotion distribution PCC:", test_emopcc.item(),
      "\n", "==========================")

import csv

with open("GC_POSITIVE.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(GC_est)

