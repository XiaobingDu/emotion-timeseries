
# -*-coding:utf-8-*-

from __future__ import print_function
import torch
from model_co_attn_GC import MovieNet
from dataManager import dataSplit, get_sample_data
# from utils_co_attn_GC import adjust_learning_rate, MediaEvalDataset, prsn, save_ckp, load_ckp, AveragePrecisionMeter
from utils_co_attn_GC import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
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

path1 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_9_DOM/'
path2 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/DOM_featureAll.mat'
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
args['out_layer'] = 2048 #9
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
num_epochs = 100 #20
batch_size = 32
lr =1e-4
GC_est =None
# 对应5个脑区的电极idx：Frontal、Temporal、Central、Parietal、Occipital
idx = [[0 ,1 ,2 ,3 ,4 ,5 ,6] ,[7 ,11 ,12 ,16 ,17 ,21 ,22 ,26] ,[8 ,9 ,10 ,13 ,14 ,15 ,18 ,19 ,20] ,[23 ,24 ,25]
       ,[27 ,28 ,29]]

# load train, val, test data
data_set = get_sample_data(path1 ,path2)
train_data, val_data, test_data, train_dis, val_dis, test_dis, train_dom_label, val_dom_label, test_dom_label = dataSplit(path1 ,data_set ,db_name)

#通过 MediaEvalDataset 将数据进行加载，返回Dataset对象，包含data和labels
trSet = MediaEvalDataset(train_data, train_dis, train_dom_label, idx)
valSet = MediaEvalDataset(val_data, val_dis, val_dom_label, idx)
testSet = MediaEvalDataset(test_data, test_dis, test_dom_label, idx)

#读取数据
trDataloader = DataLoader(trSet ,batch_size=batch_size ,shuffle=True ,num_workers=0) # len = 5079 (batches)
valDataloader = DataLoader(valSet ,batch_size=batch_size ,shuffle=True ,num_workers=0) #len = 3172
testDataloader = DataLoader(testSet ,batch_size=batch_size ,shuffle=True ,num_workers=0) #len = 2151

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
kl_div = torch.nn.KLDivLoss(size_average = False, reduce = True)
#from multi-label dom_emotion predict
MLSML = torch.nn.MultiLabelSoftMarginLoss()

# measure mAP
difficult_examples = False
ap = AveragePrecisionMeter(difficult_examples)

def on_start_batch(target_gt):
    target_gt = target_gt
    return target_gt

def on_end_batch(AveragePrecisionMeter, epoch_num, output, target_gt, multiLabel_loss, state = 'training'):
    # measure mAP
    AveragePrecisionMeter.add(output, target_gt)

    if state == 'training':
        print('Epoch: [{0}]\t'
              'Traning: \t MultiLabel-Loss {loss:.4f}\t'.format(epoch_num, loss=multiLabel_loss))
    elif state == 'validation':
        print('Validation: \t MultiLabel-Loss {loss:.4f}'.format(loss=multiLabel_loss))
    elif state == 'test':
        print('Test: \t MultiLabel-Loss {loss:.4f}'.format(loss=multiLabel_loss))



def on_start_epoch(AveragePrecisionMeter):
    AveragePrecisionMeter.reset()

def on_end_epoch(AveragePrecisionMeter, epoch_num, multiLabel_loss, state = 'training'):
    map = 100 * AveragePrecisionMeter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = AveragePrecisionMeter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = AveragePrecisionMeter.overall_topk(3)

    if state == 'training':
        print('Training: \t Epoch: [{0}]\t'
              'Loss {loss:.4f}\t'
              'mAP {map:.3f}'.format(epoch_num, loss=multiLabel_loss, map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    elif state == 'validation':
        print('Validation: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=multiLabel_loss, map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
    elif state == 'test':
        print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=multiLabel_loss, map=map))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

    return map

#traning&val&test
for epoch_num in range(num_epochs):
    #    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_learning_rate(optimizer, epoch_num, lr)

    #start_epoch
    on_start_epoch(ap)

    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train()  # the state of model
    # Variables to track training performance:
    avg_tr_loss = 0
    for i, data in enumerate(trDataloader):
        # print("Training .... 第 {} 个Batch.....".format(i))
        st_time = time.time()
        train, dis, dom_label, Frontal, Temporal, Central, Parietal, Occipital = data  # get training date

        if args['use_cuda']: # use cuda
            train = torch.nn.Parameter(train).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            dom_label = torch.nn.Parameter(dom_label.float()).cuda()
            Frontal = torch.nn.Parameter(Frontal).cuda()
            Temporal = torch.nn.Parameter(Temporal).cuda()
            Central = torch.nn.Parameter(Central).cuda()
            Parietal = torch.nn.Parameter(Parietal).cuda()
            Occipital = torch.nn.Parameter(Occipital).cuda()

        train.requires_grad_()  # backward
        dis.requires_grad_()
        dom_label.requires_grad_()
        Frontal.requires_grad_()
        Temporal.requires_grad_()
        Central.requires_grad_()
        Parietal.requires_grad_()
        Occipital.requires_grad_()

        # start_batch
        target_gt = on_start_batch(dom_label)

        # Forward pass
        emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 \
            = net(train, Frontal, Temporal, Central, Parietal, Occipital, dis)
        #get GC
        train_model_gista(shared_encoder, input_clstm, lam=0.5, lam_ridge=1e-4, lr=0.001, max_iter=1, check_every=1000, truncation=64)
        GC_est = shared_encoder.GC().cpu().data.numpy()

        emot_dis = emot_dis.squeeze(dim=0)
        dis = torch.squeeze(dis,dim=1)

        # mamx-min norm
        emot_dis = (2*(emot_dis - torch.min(emot_dis))/(torch.max(emot_dis) - torch.min(emot_dis))) -1
        # mse loss
        # l = mse(emot_score[:,0].unsqueeze(dim=1), labels1) + mse(emot_score[:,1].unsqueeze(dim=1), labels2)
        # kldiv loss
        emot_dis = torch.tensor(emot_dis, dtype=torch.double)  #<0

        #emotion distribution loss
        dis = torch.tensor(dis, dtype=torch.double) #>0
        softmax = torch.nn.Softmax(dim=1)
        dis = softmax(dis)
        loss1 = kl_div(emot_dis, dis)
        loss1 = Variable(loss1, requires_grad=True)

        #multi-labe emotion prediction loss
        target_gt = torch.tensor(target_gt, dtype=torch.double)
        loss2 = MLSML(emot_dis.cuda(), target_gt.cuda())
        loss2 = Variable(loss2, requires_grad=True)

        loss = loss1 + loss2
        loss = Variable(loss, requires_grad=True)

        # Backprop and update weights
        optimizer.zero_grad()
        loss.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        avg_tr_loss += loss.item()
        if i % 100 == 0:
            #end_batch
            on_end_batch(ap, epoch_num+1, emot_dis, target_gt, loss2, state= 'training')
    #end_epoch
    on_end_epoch(ap,epoch_num+1, loss2, state= 'training')

    #emotion distribution metrics
    # euclidean
    euclidean = euclidean_dist(dis.shape[0], dis, emot_dis)
    # chebyshev
    chebyshev = chebyshev_dist(dis.shape[0], dis, emot_dis)
    # Kullback-Leibler divergence
    kldist = KL_dist(dis, emot_dis)
    # clark
    clark = clark_dist(dis, emot_dis)
    # canberra
    canberra = canberra_dist(dis, emot_dis)
    # cosine
    cosine = cosine_dist(dis, emot_dis)
    # intersection
    intersection = intersection_dist(dis, emot_dis)

    # print(GC_est)
    print("Epoch no:" ,epoch_num +1, "| Avg train loss:" ,format(avg_tr_loss /len(trSet) ,'0.4f') )
    print('euclidean_dist: {euclidean_dist:.4f}\t'
          'chebyshev_dist: {chebyshev_dist:.4f}\t'
          'kldist: {kldist:.4f}\t'
          'clark_dist: {clark_dist:.4f}\t'
          'canberra_dist: {canberra_dist:.4f}\t'
          'cosine_dist: {cosine_dist:.4f}\t'
          'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
                                                                chebyshev_dist=chebyshev, kldist=kldist,
                                                                clark_dist=clark, canberra_dist=canberra,
                                                                cosine_dist=cosine,
                                                                intersection_dist=intersection))
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.eval()
    val_kl = 0
    emopcc = 0

    # start_epoch
    on_start_epoch(ap)

    for i, data in enumerate(valDataloader):
        # print("Val ..... 第 {} 个Batch.....".format(i))
        st_time = time.time()
        val, dis, dom_label, Frontal, Temporal, Central, Parietal, Occipital = data

        if args['use_cuda']:
            val = torch.nn.Parameter(val).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            dom_label = torch.nn.Parameter(dom_label.float()).cuda()
            Frontal = torch.nn.Parameter(Frontal).cuda()
            Temporal = torch.nn.Parameter(Temporal).cuda()
            Central = torch.nn.Parameter(Central).cuda()
            Parietal = torch.nn.Parameter(Parietal).cuda()
            Occipital = torch.nn.Parameter(Occipital).cuda()

        # start_batch
        target_gt = on_start_batch(dom_label)

        # Forward pass
        emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
            net(val, Frontal, Temporal, Central, Parietal, Occipital, dis)

        emot_dis = emot_dis.squeeze(dim=0)
        dis = torch.squeeze(dis,dim=1)

        #min-max norm
        emot_dis = (2*(emot_dis - torch.min(emot_dis))/(torch.max(emot_dis) - torch.min(emot_dis))) -1
        # 每一个batch的average mse loss相加
        # valmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
        # aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]
        emot_dis = torch.tensor(emot_dis, dtype=torch.double) #[32,9]
        #emotion distribution loss
        dis = torch.tensor(dis, dtype=torch.double)
        loss1 = kl_div(emot_dis, dis)
        #multi-label emotion prediction loss
        target_gt = torch.tensor(target_gt, dtype=torch.double)
        loss2 = MLSML(emot_dis.cuda(), target_gt.cuda())
        val_loss = loss1 + loss2
        val_loss += val_loss /dis.shape[0]

        if i % 100 == 0:
            #measure mAP
            #end_batch
            on_end_batch(ap, epoch_num+1, emot_dis, target_gt, loss2, state='validation')

        # Pearson correlation
        emopcc += pearsonr(emot_dis.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]

    # 每一个epoch loss平均
    epoch_loss = val_loss /len(valSet)
    # 每一个epoch pcc平均
    epoch_pcc = emopcc / len(valSet)
    # validation loss
    val_loss = epoch_loss

    #end_epoch
    on_end_epoch(ap, epoch_num+1, loss2, state='validation')

    # emotion distribution metrics
    # euclidean
    euclidean = euclidean_dist(dis.shape[0], dis, emot_dis)
    # chebyshev
    chebyshev = chebyshev_dist(dis.shape[0], dis, emot_dis)
    # Kullback-Leibler divergence
    kldist = KL_dist(dis, emot_dis)
    # clark
    clark = clark_dist(dis, emot_dis)
    # canberra
    canberra = canberra_dist(dis, emot_dis)
    # cosine
    cosine = cosine_dist(dis, emot_dis)
    # intersection
    intersection = intersection_dist(dis, emot_dis)

    print("Validation: Epoch emotion distribution KLDivLoss:", epoch_loss.item() , "\nEpoch emotion distribution PCC:", epoch_pcc.item() ,"\n", "==========================")
    print('euclidean_dist: {euclidean_dist:.4f}\t'
          'chebyshev_dist: {chebyshev_dist:.4f}\t'
          'kldist: {kldist:.4f}\t'
          'clark_dist: {clark_dist:.4f}\t'
          'canberra_dist: {canberra_dist:.4f}\t'
          'cosine_dist: {cosine_dist:.4f}\t'
          'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
                                                                chebyshev_dist=chebyshev, kldist=kldist,
                                                                clark_dist=clark, canberra_dist=canberra,
                                                                cosine_dist=cosine,
                                                                intersection_dist=intersection))
    # checkpoint
    checkpoint = {
        'epoch': epoch_num + 1,
        'valid_loss_min_kl': epoch_loss,
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


# testing
net = MovieNet(args)
net, optimizer, start_epoch, valid_loss_min_kl = load_ckp(
    best_model_path + "/train_co_attn_GC_best_model.pt", net, optimizer)
# testing
net.eval()
test_kl = 0
emopcc = 0

# start_epoch
on_start_epoch(ap)

for i, data in enumerate(testDataloader):
    st_time = time.time()
    test, dis, dom_label, Frontal, Temporal, Central, Parietal, Occipital = data
    dis = dis

    if args['use_cuda']:
        test = torch.nn.Parameter(test).cuda()
        dis = torch.nn.Parameter(dis).cuda()
        dom_label = torch.nn.Parameter(dom_label.float()).cuda()
        Frontal = torch.nn.Parameter(Frontal).cuda()
        Temporal = torch.nn.Parameter(Temporal).cuda()
        Central = torch.nn.Parameter(Central).cuda()
        Parietal = torch.nn.Parameter(Parietal).cuda()
        Occipital = torch.nn.Parameter(Occipital).cuda()

    # start batch
    target_gt = on_start_batch(dom_label)

    # Forward pass
    emot_dis, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
        net(test, Frontal, Temporal, Central, Parietal, Occipital, dis)
    # print(att_1, att_2, att_3, att_4, att_5, att_6)
    emot_dis = emot_dis.squeeze(dim=0)
    dis = torch.squeeze(dis, dim=1)

    # min-max norm
    emot_dis = (2*(emot_dis - torch.min(emot_dis))/(torch.max(emot_dis) - torch.min(emot_dis))) -1
    # mse loss
    # testmse += mse(emot_score[:, 0].unsqueeze(dim=1), labels1)/labels1.shape[0]
    # aromse += mse(emot_score[:, 1].unsqueeze(dim=1), labels2)/labels2.shape[0]
    # kldiv loss
    emot_dis = torch.tensor(emot_dis, dtype=torch.double)  # [32,9]

    #emotion distribution loss
    dis = torch.tensor(dis, dtype=torch.double)
    loss1 = kl_div(emot_dis, dis)
    #multi-label emotion predictation loss
    target_gt = torch.tensor(target_gt, dtype=torch.double)
    loss2 = MLSML(emot_dis.cuda(), target_gt.cuda())

    test_loss = loss1 + loss2
    test_loss += test_loss / dis.shape[0]

    if i % 100 == 0:
        # measure mAP
        on_end_batch(ap, epoch_num+1, emot_dis, target_gt, loss2, state= 'test')

    # pearson correlation
    emopcc += pearsonr(emot_dis.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
# average loss
test_testkl = test_loss / len(testSet)
# average pcc
test_emopcc = emopcc / len(testSet)

on_end_epoch(ap, epoch_num+1, loss2, state= 'test')

# emotion distribution metrics
# euclidean
euclidean = euclidean_dist(dis.shape[0], dis, emot_dis)
# chebyshev
chebyshev = chebyshev_dist(dis.shape[0], dis, emot_dis)
# Kullback-Leibler divergence
kldist = KL_dist(dis, emot_dis)
# clark
clark = clark_dist(dis, emot_dis)
# canberra
canberra = canberra_dist(dis, emot_dis)
# cosine
cosine = cosine_dist(dis, emot_dis)
# intersection
intersection = intersection_dist(dis, emot_dis)

print("Test Emotion distribution KLDivLoss:", test_testkl.item(), "\Test Emotion distribution PCC:", test_emopcc.item(),
      "\n", "==========================")
print('euclidean_dist: {euclidean_dist:.4f}\t'
          'chebyshev_dist: {chebyshev_dist:.4f}\t'
          'kldist: {kldist:.4f}\t'
          'clark_dist: {clark_dist:.4f}\t'
          'canberra_dist: {canberra_dist:.4f}\t'
          'cosine_dist: {cosine_dist:.4f}\t'
          'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
                                                                chebyshev_dist=chebyshev, kldist=kldist,
                                                                clark_dist=clark, canberra_dist=canberra,
                                                                cosine_dist=cosine,
                                                                intersection_dist=intersection))

print(att_1, att_2, att_3, att_4, att_5, att_6)

import csv

with open("GC_POSITIVE.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(GC_est)

