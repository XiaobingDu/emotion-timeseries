# -*-coding:utf-8-*-

from __future__ import print_function
from model import EEGEncoder
from dataManager import five_fold, dataSplit, get_sample_data, get_sample_data_withoutOverlap
from utils import *
from torch.utils.data import DataLoader
from scipy.stats.mstats import pearsonr
import time, argparse
import codecs
from multilabelMetrics.examplebasedclassification import *
from multilabelMetrics.examplebasedranking import *
from multilabelMetrics.labelbasedclassification import *
from multilabelMetrics.labelbasedranking import *
import warnings

warnings.filterwarnings('ignore')
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()

parser.add_argument('--path1', type=str, choices=[
    "/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_addLabel_sum1/",
    '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_9_DOM/',
    '../EEG_PSD_9_DOM/'])
parser.add_argument('--path2', type=str, choices=[
    "/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/featureAll.mat",
    '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/DOM_featureAll.mat',
    '../DOM_feature_all/DOM_featureAll.mat'])
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--learning_rate', type=float, default= 1e-3, # 0.001
                    help='Initial learning rate.')
parser.add_argument('--iter_num', type=int,
                    help='Number of iterate to train.')
parser.add_argument('--lamda', type=float, default=0.6,
                    help='The lamda is the weight to control the trade-off between two type losses.')
parser.add_argument('--overlap', type=str, default='with', choices=['with', 'without'],
                    help='Get the samples with/without time overlap.')
parser.add_argument('--sub_id', type=int, default=0,
                    help='The subject ID for Test.')
parser.add_argument('--fold_id', type=int, default=1,
                    help='The fold id  for Test.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--label_num', type=int, default=9)
parser.add_argument('--db_name', type=str, default='LDL_data')
parser.add_argument('--strategy', type=str, default='five_fold', choices=['split', 'five_fold', 'ldl_loso'])
parser.add_argument('--save_file', type=str, default='co_attn_GC')
parser.add_argument('--log_dir', type=str)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--GCN_hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--LSTM_layers', type=int, default=2,
                    help='Number of LSTM layers.')
parser.add_argument('--LSTM_hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--attn_len', type=int, default=1,
                    help='attn_len = time_sequence')
parser.add_argument('--out_layer', type=int, default=9)
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
FLAGS = parser.parse_args()

args = {}
path1 = FLAGS.path1
path2 = FLAGS.path2
db_name = 'LDL_data'
best_model_path = "./best_model"
checkpoint_path = "./checkpoints"
valid_loss_min = np.Inf
epoch_cosine_min = np.NINF
epoch_accuracy_min = np.NINF

# Network Arguments
args['time_steps'] = 30  # timesteps
args['feature_dim'] = 75  # 15channels * 5
args['out_layer'] = FLAGS.out_layer
args['channels'] = 30  # channel
args['feature_len'] = 150  # 30*5 = 150
args['enc_dim'] = 256
args['hidden_dim'] = 256
args['attn_len'] = FLAGS.attn_len
args['dropout_prob'] = FLAGS.dropout
args['use_cuda'] = True
args['train_flag'] = True
args['optimizer'] = 'rmsprop'# 'adam'

num_epochs = FLAGS.epochs
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
lamda = FLAGS.lamda
overlap = FLAGS.overlap
strategy = FLAGS.strategy
fold_id = FLAGS.fold_id
GC_est = None
# 对应左右脑区的电极idx：left, right
idx = [[0, 2, 3, 4, 7, 8, 12, 13, 14, 17, 18, 22, 23, 24, 27], [1, 5, 6, 9, 10, 11, 15, 16, 19, 20, 21, 25, 26, 28, 29]]

# load train, val, test data
if overlap == 'with':
    print('------overlap: True------')
    data_set = get_sample_data(path1, path2)
elif overlap == 'without':
    print('------overlap: False------')
    data_set = get_sample_data_withoutOverlap(path1, path2)

if strategy == 'five_fold':
    print('------strategy: five_fold------')
    train_data, val_data, train_dis, val_dis, train_dom_label, val_dom_label = five_fold(data_set, fold_id, db_name)
    test_data = val_data
    test_dis = val_dis
    test_dom_label = val_dom_label
elif strategy == 'split':
    print('------strategy: split------')
    train_data, val_data, test_data, train_dis, val_dis, test_dis, train_dom_label, val_dom_label, test_dom_label = dataSplit(
        path1, data_set, db_name)

train_data = train_data.astype(np.float32)
val_data = val_data.astype(np.float32)
test_data = test_data.astype(np.float32)

train_dis = train_dis.astype(np.float32)
val_dis = val_dis.astype(np.float32)
test_dis = test_dis.astype(np.float32)

train_dom_label = train_dom_label.astype(np.float32)
val_dom_label = val_dom_label.astype(np.float32)
test_dom_label = test_dom_label.astype(np.float32)

# 通过 MediaEvalDataset 将数据进行加载，返回Dataset对象，包含data和labels
trSet = MediaEvalDataset(train_data, train_dis, train_dom_label, idx)
valSet = MediaEvalDataset(val_data, val_dis, val_dom_label, idx)
testSet = MediaEvalDataset(test_data, test_dis, test_dom_label, idx)

# 读取数据
trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=0)  # len = 5079 (batches)
valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=0)  # len = 3172
testDataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=0)  # len = 2151

# Initialize network
net = EEGEncoder(args)
if args['use_cuda']:
    net = net.cuda()

## Initialize optimizer
if args['optimizer'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-4)
elif args['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4) # 1e-8
elif args['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)

kl_div = torch.nn.KLDivLoss(size_average=True, reduce=True)
BCE = torch.nn.BCEWithLogitsLoss()

# write in file and save
result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
if FLAGS.strategy == "ldl_loso":
    result.write('========\nsubject %s\n========\n' % str(FLAGS.sub_id))
elif FLAGS.strategy == "five_fold":
    result.write('\n========\nfold %s\n========\n' % str(FLAGS.fold_id))
    result.write('\n========\nlamda %s\n========\n' % str(FLAGS.lamda))
elif FLAGS.strategy == "split":
    result.write('\n========\nsplit %s\n========\n' % '5:3:2')
    result.write('\n========\nlamda %s\n========\n' % str(FLAGS.lamda))
result.write('model parameters: %s' % str(FLAGS))
result.close()

# traning&val&test
epoch_euclidean = 0
epoch_chebyshev = 0
epoch_kldist = 0
epoch_clark = 0
epoch_canberra = 0
epoch_cosine = 0
epoch_intersection = 0
epoch_hammingLoss = 0
epoch_eb_accuracy = 0
epoch_eb_precision = 0
epoch_eb_recall = 0
epoch_eb_fbeta = 0
epoch_oneError = 0
epoch_coverage = 0
epoch_averagePrecision = 0
epoch_rankingLoss = 0
epoch_accuracyMacro = 0
epoch_fbetaMicro = 0
epoch_accuracyMicro = 0
epoch_precisionMacro = 0
epoch_precisionMicro = 0
epoch_recallMacro = 0
epoch_recallMicro = 0
epoch_fbetaMacro = 0

for epoch_num in range(num_epochs):
    #    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # adjust_learning_rate(optimizer, epoch_num, lr)
    scheduler.step()
    print("第%d个epoch的学习率：%f" % (epoch_num, optimizer.param_groups[0]['lr']))

    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train()  # the state of model
    # Variables to track training performance:
    avg_tr_loss = 0
    for i, data in enumerate(trDataloader):
        print("Training .... 第 {} 个Batch.....".format(i + 1))
        st_time = time.time()
        train, dis, dom_label, left, right = data  # get training date

        if args['use_cuda']:  # use cuda
            train = torch.nn.Parameter(train).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            dom_label = torch.nn.Parameter(dom_label.float()).cuda()
            left = torch.nn.Parameter(left).cuda()
            right = torch.nn.Parameter(right).cuda()

        train.requires_grad_()  # backward
        dis.requires_grad_()
        dom_label.requires_grad_()
        left.requires_grad_()
        right.requires_grad_()

        # Forward pass
        predict, time_att, channel_att = net(train, left, right, dis)

        # for loss1
        # softmax layer
        softmax = torch.nn.Softmax(dim=1)
        dis_prediction = softmax(predict)
        dis_prediction = dis_prediction.squeeze(dim=0)  # [32,9]
        dis = torch.squeeze(dis, dim=1)
        # loss1: KLDivLoss
        loss1 = kl_div(dis_prediction.log(), dis)
        # for loss2
        predict_sig = torch.sigmoid(predict)
        # loss2: BCELoss
        target_gt = dom_label
        target_gt = target_gt.detach()
        loss2 = BCE(predict, target_gt)
        # loss2:
        loss = lamda * loss1 + (1 - lamda) * loss2
        # print('loss1.....:', loss1.item())
        # print('loss2......:', loss2.item())
        # print('loss.......:', loss.item())

        # Backprop and update weights
        optimizer.zero_grad()
        loss.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()


        avg_tr_loss = loss.item()
        avg_tr_loss += avg_tr_loss / dis.shape[0]
        # avg_tr_loss += loss.item()

        # emotion distribution metrics
        # euclidean
        euclidean = euclidean_dist(dis.shape[0], dis, dis_prediction)
        # chebyshev
        chebyshev = chebyshev_dist(dis.shape[0], dis, dis_prediction)
        # Kullback-Leibler divergence
        kldist = KL_dist(dis, dis_prediction)
        # clark
        clark = clark_dist(dis, dis_prediction)
        # canberra
        canberra = canberra_dist(dis, dis_prediction)
        # cosine
        cosine = cosine_dist(dis, dis_prediction)
        # intersection
        intersection = intersection_dist(dis, dis_prediction)

        # for multilabel prediction
        # # example-based-classification
        train_subsetAccuracy = subsetAccuracy(dom_label, predict_sig)
        train_hammingLoss = hammingLoss(dom_label, predict_sig)
        train_eb_accuracy = accuracy(dom_label, predict_sig)
        train_eb_precision = precision(dom_label, predict_sig)
        train_eb_recall = recall(dom_label, predict_sig)
        train_eb_fbeta = fbeta(dom_label, predict_sig)
        #
        # # example-based-ranking
        train_oneError = oneError(dom_label, predict_sig)
        train_coverage = coverage(dom_label, predict_sig)
        train_averagePrecision = averagePrecision(dom_label, predict_sig)
        train_rankingLoss = rankingLoss(dom_label, predict_sig)
        #
        # # label-based-classification
        train_accuracyMacro = accuracyMacro(dom_label, predict_sig)
        train_accuracyMicro = accuracyMicro(dom_label, predict_sig)
        train_precisionMacro = precisionMacro(dom_label, predict_sig)
        train_precisionMicro = precisionMicro(dom_label, predict_sig)
        train_recallMacro = recallMacro(dom_label, predict_sig)
        train_recallMicro = recallMicro(dom_label, predict_sig)
        train_fbetaMacro = fbetaMacro(dom_label, predict_sig)
        train_fbetaMicro = fbetaMicro(dom_label, predict_sig)

        # results
        if i % 10 == 0:
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

            print("Training set results:\n",
                  'Multilabel metrics: example-based-classification:\n',
                  "subsetAccuracy= {:.4f}".format(train_subsetAccuracy),
                  "hammingLoss= {:.4f}".format(train_hammingLoss),
                  "accuracy= {:.4f}".format(train_eb_accuracy),
                  "precision= {:.4f}".format(train_eb_precision),
                  "recall= {:.4f}".format(train_eb_recall),
                  "f-1= {:.4f}".format(train_eb_fbeta)
                  )

            print("Training set results:\n",
                  'Multilabel metrics: example-based-ranking:\n',
                  "oneError= {:.4f}".format(train_oneError),
                  "converage= {:.4f}".format(train_coverage),
                  "averagePrecision= {:.4f}".format(train_averagePrecision),
                  "rankingLoss= {:.4f}".format(train_rankingLoss))

            print("Training set results:\n",
                  'Multilabel metrics: label-based-classification:\n',
                  "accuracyMacro= {:.4f}".format(train_accuracyMacro),
                  "accuracyMicro= {:.4f}".format(train_accuracyMicro),
                  "precisionMacro= {:.4f}".format(train_precisionMacro),
                  "precisionMicro= {:.4f}".format(train_precisionMicro),
                  "recallMacro= {:.4f}".format(train_recallMacro),
                  "recallMicro= {:.4f}".format(train_recallMicro),
                  "f-1Macro= {:.4f}".format(train_fbetaMacro),
                  "f-1Micro= {:.4f}".format(train_fbetaMicro))

            result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
    #         result.write("\n------------------------------------------------------------------\n")
    #         result.write("Training GC_est:\n")
    #         result.write('%s\n' % GC_est)
    #         result.write('Training co-attention:\n ')
    #         result.write('att_1: \t')
    #         result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
    #
    #         result.write('\n Epoch: [{0}]: Training....\n'.format(epoch_num + 1))
    #         result.write('epoch_num: {epoch_num:.1f}\t'
    #                      'learning_rate: {learning_rate:.6f}\t'.format(epoch_num=epoch_num, learning_rate=optimizer.param_groups[0]['lr']))
    #         result.write("\n========================================\n")
    #         result.write('euclidean_dist: {euclidean_dist:.4f}\t'
    #                      'chebyshev_dist: {chebyshev_dist:.4f}\t'
    #                      'kldist: {kldist:.4f}\t'
    #                      'clark_dist: {clark_dist:.4f}\t'
    #                      'canberra_dist: {canberra_dist:.4f}\t'
    #                      'cosine_dist: {cosine_dist:.4f}\t'
    #                      'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
    #                                                                            chebyshev_dist=chebyshev, kldist=kldist,
    #                                                                            clark_dist=clark, canberra_dist=canberra,
    #                                                                            cosine_dist=cosine,
    #                                                                            intersection_dist=intersection))
    #         result.write("\n------------------------------------------------------------------\n")
    #         result.write("Training set results:\n")
    #         result.write('Multilabel metrics: example-based-classification:\n')
    #         result.write(
    #             'subsetAccuracy: {subsetAccuracy:.4f}\t'
    #             'hammingLoss: {hammingLoss:.4f}\t'
    #             'accuracy: {accuracy:.4f}\t'
    #             'precision: {precision:.4f}\t'
    #             'recall: {recall:.4f}\t'
    #             'fbeta: {fbeta:.4f}\t'.format(
    #                 subsetAccuracy=train_subsetAccuracy,
    #                 hammingLoss=train_hammingLoss,
    #                 accuracy=train_eb_accuracy,
    #                 precision=train_eb_precision,
    #                 recall=train_eb_recall, fbeta=train_eb_fbeta)
    #         )
    #
    #         result.write("\n")
    #         result.write('Multilabel metrics: example-based-ranking:\n')
    #         result.write('oneError: {oneError:.4f}\t'
    #                      'averagePrecision: {averagePrecision:.4f}\t'
    #                      'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=train_oneError,
    #                                                                averagePrecision=train_averagePrecision,
    #                                                                rankingLoss=train_rankingLoss))
    #
    #         result.write("\n")
    #         result.write('Multilabel metrics: label-based-classification:\n')
    #         result.write('accuracyMacro: {accuracyMacro:.4f}\t'
    #                      'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=train_accuracyMacro,
    #                                                              fbetaMicro=train_fbetaMicro))
    #
    # print("Epoch no:", epoch_num + 1, "| Avg_train_loss:".format(avg_tr_loss / len(trSet), '0.4f'))
    # result.write("\n------------------------------------------------------------------\n")
    # result.write("Epoch no: {epoch: .4f}\t"
    #              "| Avg_train_loss: {loss:.4f}\t".format(epoch=epoch_num + 1, loss=avg_tr_loss / len(trSet)))

    ## Validate:
    net.eval()
    val_kl = 0
    emopcc = 0
    val_cosine = 0
    val_accuracy = 0
    sum_euclidean = 0
    sum_chebyshev = 0
    sum_kldist = 0
    sum_clark = 0
    sum_canberra = 0
    sum_cosine = 0
    sum_intersection = 0
    sum_hammingLoss = 0
    sum_eb_accuracy = 0
    sum_eb_precision = 0
    sum_eb_recall = 0
    sum_eb_fbeta = 0
    sum_oneError = 0
    sum_averagePrecision = 0
    sum_rankingLoss = 0
    sum_accuracyMacro = 0
    sum_fbetaMicro = 0
    sum_coverage = 0
    sum_accuracyMicro = 0
    sum_precisionMacro = 0
    sum_precisionMicro = 0
    sum_recallMacro = 0
    sum_recallMicro = 0
    sum_fbetaMacro = 0
    cnt = 0

    for i, data in enumerate(valDataloader):
        cnt += 1
        print("Val ..... 第 {} 个Batch.....".format(i + 1))
        st_time = time.time()
        val, dis, dom_label, left, right = data

        if args['use_cuda']:
            val = torch.nn.Parameter(val).cuda()
            dis = torch.nn.Parameter(dis).cuda()
            dom_label = torch.nn.Parameter(dom_label.float()).cuda()
            left = torch.nn.Parameter(left).cuda()
            right = torch.nn.Parameter(right).cuda()

        # Forward pass
        predict, time_att, channel_att = net(val, left, right, dis)

        # for loss1
        softmax = torch.nn.Softmax(dim=1)
        dis_prediction = softmax(predict)
        dis_prediction = dis_prediction.squeeze(dim=0)  # [32,9]
        dis = torch.squeeze(dis, dim=1)
        loss1 = kl_div(dis_prediction.log(), dis)
        # for loss2
        predict_sig = torch.sigmoid(predict)
        # loss2: BCELoss
        dom_label = dom_label.detach()
        loss2 = BCE(predict, dom_label)
        # loss2: MLSML
        # loss2 = MLSML(predict.cuda(), target_gt.cuda())
        loss = lamda * loss1 + (1 - lamda) * loss2
        val_loss = loss.item()
        val_loss += val_loss / dis.shape[0]

        # emotion distribution metrics
        # euclidean
        euclidean = euclidean_dist(dis.shape[0], dis, dis_prediction)
        sum_euclidean += euclidean
        # chebyshev
        chebyshev = chebyshev_dist(dis.shape[0], dis, dis_prediction)
        sum_chebyshev += chebyshev
        # Kullback-Leibler divergence
        kldist = KL_dist(dis, dis_prediction)
        sum_kldist += kldist
        # clark
        clark = clark_dist(dis, dis_prediction)
        sum_clark += clark
        # canberra
        canberra = canberra_dist(dis, dis_prediction)
        sum_canberra += canberra
        # cosine
        cosine = cosine_dist(dis, dis_prediction)
        sum_cosine += cosine
        # intersection
        intersection = intersection_dist(dis, dis_prediction)
        sum_intersection += intersection

        # for multilabel prediction
        # example-based-classification
        val_hammingLoss = hammingLoss(dom_label, predict_sig)
        sum_hammingLoss += val_hammingLoss
        val_eb_accuracy = accuracy(dom_label, predict_sig)
        val_accuracy += val_eb_accuracy
        sum_eb_accuracy += val_eb_accuracy
        val_eb_precision = precision(dom_label, predict_sig)
        sum_eb_precision += val_eb_precision
        val_eb_recall = recall(dom_label, predict_sig)
        sum_eb_recall += val_eb_recall
        val_eb_fbeta = fbeta(dom_label, predict_sig)
        sum_eb_fbeta += val_eb_fbeta

        # example-based-ranking
        val_oneError = oneError(dom_label, predict_sig)
        sum_oneError += val_oneError
        val_coverage = coverage(dom_label, predict_sig)
        sum_coverage += val_coverage
        val_averagePrecision = averagePrecision(dom_label, predict_sig)
        sum_averagePrecision += val_averagePrecision
        val_rankingLoss = rankingLoss(dom_label, predict_sig)
        sum_rankingLoss += val_rankingLoss

        # label-based-classification
        val_accuracyMacro = accuracyMacro(dom_label, predict_sig)
        sum_accuracyMacro += val_accuracyMacro
        val_accuracyMicro = accuracyMicro(dom_label, predict_sig)
        sum_accuracyMicro += val_accuracyMicro
        val_precisionMacro = precisionMacro(dom_label, predict_sig)
        sum_precisionMacro += val_precisionMacro
        val_precisionMicro = precisionMicro(dom_label, predict_sig)
        sum_precisionMicro += val_accuracyMicro
        val_recallMacro = recallMacro(dom_label, predict_sig)
        sum_recallMacro += val_recallMacro
        val_recallMicro = recallMicro(dom_label, predict_sig)
        sum_recallMicro += val_recallMicro
        val_fbetaMacro = fbetaMacro(dom_label, predict_sig)
        sum_fbetaMacro += val_fbetaMacro
        val_fbetaMicro = fbetaMicro(dom_label, predict_sig)
        sum_fbetaMicro += val_fbetaMicro

        if i % 10 == 0:
            print('euclidean_dist: {euclidean_dist:.4f}\t'
                  'chebyshev_dist: {chebyshev_dist:.4f}\t'
                  'kldist: {kldist:.4f}\t'
                  'clark_dist: {clark_dist:.4f}\t'
                  'canberra_dist: {canberra_dist:.4f}\t'
                  'cosine_dist: {cosine_dist:.4f}\t'
                  'intersection_dist: {intersection_dist:.4f}\n'.format(euclidean_dist=euclidean,
                                                                        chebyshev_dist=chebyshev, kldist=kldist,
                                                                        clark_dist=clark, canberra_dist=canberra,
                                                                        cosine_dist=cosine,
                                                                        intersection_dist=intersection))

            print("Val set results:\n",
                  'Multilabel metrics: example-based-classification:\n',
                  "hammingLoss= {:.4f}".format(val_hammingLoss),
                  "accuracy= {:.4f}".format(val_eb_accuracy),
                  "precision= {:.4f}".format(val_eb_precision),
                  "recall= {:.4f}".format(val_eb_recall),
                  "fbeta= {:.4f}".format(val_eb_fbeta))

            print("Val set results:\n",
                  'Multilabel metrics: example-based-ranking:\n',
                  "oneError= {:.4f}".format(val_oneError),
                  "coverage= {:.4f}".format(val_coverage),
                  "averagePrecision= {:.4f}".format(val_averagePrecision),
                  "rankingLoss= {:.4f}".format(val_rankingLoss))

            print("Val set results:\n",
                  'Multilabel metrics: label-based-classification:\n',
                  "accuracyMacro= {:.4f}".format(val_accuracyMacro),
                  "accuracyMicro= {:.4f}".format(val_accuracyMicro),
                  "precisionMacro= {:.4f}".format(val_precisionMacro),
                  "precisionMicro= {:.4f}".format(val_precisionMicro),
                  "recallMacro= {:.4f}".format(val_recallMacro),
                  "recallMicro= {:.4f}".format(val_recallMicro),
                  "f-1Macro= {:.4f}".format(val_fbetaMacro),
                  "f-1Micro= {:.4f}".format(val_fbetaMicro))

            # result.write("\n------------------------------------------------------------------\n")
            # result.write('Val co-attention:\n ')
            # result.write('att_1: \t')
            # result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
            #
            # result.write('\n Epoch: [{0}]: Test....\n'.format(epoch_num + 1))
            # result.write("\n========================================\n")
            # result.write('euclidean_dist: {euclidean_dist:.4f}\t'
            #              'chebyshev_dist: {chebyshev_dist:.4f}\t'
            #              'kldist: {kldist:.4f}\t'
            #              'clark_dist: {clark_dist:.4f}\t'
            #              'canberra_dist: {canberra_dist:.4f}\t'
            #              'cosine_dist: {cosine_dist:.4f}\t'
            #              'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
            #                                                                    chebyshev_dist=chebyshev, kldist=kldist,
            #                                                                    clark_dist=clark, canberra_dist=canberra,
            #                                                                    cosine_dist=cosine,
            #                                                                    intersection_dist=intersection))
            # result.write("\n------------------------------------------------------------------\n")
            # result.write("Val set results:\n")
            # result.write('Multilabel metrics: example-based-classification:\n')
            # result.write(
            #     'hammingLoss: {hammingLoss:.4f}\t'
            #     'accuracy: {accuracy:.4f}\t'
            #     'precision: {precision:.4f}\t'
            #     'recall: {recall:.4f}\t'
            #     'fbeta: {fbeta:.4f}\t'.format(
            #         hammingLoss=val_hammingLoss,
            #         accuracy=val_eb_accuracy, precision=val_eb_precision,
            #         recall=val_eb_recall, fbeta=val_eb_fbeta))
            # result.write("\n")
            # result.write('Multilabel metrics: example-based-ranking:\n')
            # result.write('oneError: {oneError:.4f}\t'
            #              'coverage: {coverage:.4f}\t'
            #              'averagePrecision: {averagePrecision:.4f}\t'
            #              'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=val_oneError,
            #                                                        coverage=val_coverage,
            #                                                        averagePrecision=val_averagePrecision,
            #                                                        rankingLoss=val_rankingLoss))
            # result.write("\n")
            # result.write('Multilabel metrics: label-based-classification:\n')
            # result.write('accuracyMacro: {accuracyMacro:.4f}\t'
            #              'accuracyMicro: {accuracyMicro:.4f}\t'
            #              'precisionMacro: {precisionMacro:.4f}\t'
            #              'precisionMicro: {precisionMicro:.4f}\t'
            #              'recallMacro: {recallMacro:.4f}\t'
            #              'recallMicro: {recallMicro:.4f}\t'
            #              'fbetaMacro: {fbetaMacro:.4f}\t'
            #              'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=val_accuracyMacro,
            #                                                      accuracyMicro=val_accuracyMicro,
            #                                                      precisionMacro=val_precisionMacro,
            #                                                      precisionMicro=val_precisionMicro,
            #                                                      recallMacro=val_precisionMacro,
            #                                                      recallMicro=val_precisionMicro,
            #                                                      fbetaMacro=val_fbetaMacro,
            #                                                      fbetaMicro=val_fbetaMicro))
            #
            # # pearson correlation
            # emopcc += pearsonr(dis_prediction.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
            #
            # result.write("\n------------------------------------------------------------------\n")
            # result.write('Validation co-attention:\n ')
            # result.write('att_1: \t')
            # result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))

    # average loss
    val_testkl = val_loss / len(testSet)
    # average pcc
    val_emopcc = emopcc / len(testSet)
    ave_euclidean = sum_euclidean / cnt
    epoch_euclidean += ave_euclidean
    ave_chebyshev = sum_chebyshev / cnt
    epoch_chebyshev += ave_chebyshev
    ave_kldist = sum_kldist / cnt
    epoch_kldist += ave_kldist
    ave_clark = sum_clark / cnt
    epoch_clark += ave_clark
    ave_canberra = sum_canberra / cnt
    epoch_canberra += ave_canberra
    ave_cosine = sum_cosine / cnt
    epoch_cosine += ave_cosine
    ave_intersection = sum_intersection / cnt
    epoch_intersection += ave_intersection
    ave_hammingLoss = sum_hammingLoss / cnt
    epoch_hammingLoss += ave_hammingLoss
    ave_eb_accuracy = sum_eb_accuracy / cnt
    epoch_eb_accuracy += ave_eb_accuracy
    ave_eb_precision = sum_eb_precision / cnt
    epoch_eb_precision += ave_eb_precision
    ave_eb_recall = sum_eb_recall / cnt
    epoch_eb_recall += ave_eb_recall
    ave_eb_fbeta = sum_eb_fbeta / cnt
    epoch_eb_fbeta += ave_eb_fbeta
    ave_oneError = sum_oneError / cnt
    epoch_oneError += ave_oneError
    ave_averagePrecision = sum_averagePrecision / cnt
    epoch_averagePrecision += ave_averagePrecision
    ave_rankingLoss = sum_rankingLoss / cnt
    epoch_rankingLoss += ave_rankingLoss
    ave_accuracyMacro = sum_accuracyMacro / cnt
    epoch_accuracyMacro += ave_accuracyMacro
    ave_fbetaMicro = sum_fbetaMicro / cnt
    epoch_fbetaMicro += ave_fbetaMicro
    ave_coverage = sum_coverage / cnt
    epoch_coverage += ave_coverage
    ave_accuracyMicro = sum_accuracyMicro / cnt
    epoch_accuracyMicro += ave_accuracyMicro
    ave_precisionMacro = sum_precisionMacro / cnt
    epoch_precisionMacro += ave_precisionMacro
    ave_precisionMicro = sum_precisionMicro / cnt
    epoch_precisionMicro += ave_precisionMicro
    ave_recallMacro = sum_recallMacro / cnt
    epoch_recallMacro += ave_recallMacro
    ave_recallMicro = sum_recallMicro / cnt
    epoch_recallMicro += ave_recallMicro
    ave_fbetaMacro = sum_fbetaMacro / cnt
    epoch_fbetaMacro += ave_fbetaMacro

    result.write("\n================================================================================\n")
    result.write('Epoch: {epoch:.1f}\t'
                 'Val epoch_euclidean: {epoch_euclidean:.4f}\t'
                 'Val epoch_chebyshev: {epoch_chebyshev:.4f}\t'
                 'Val epoch_kldist: {epoch_kldist:.4f}\t'
                 'Val epoch_clark_dist: {epoch_clark_dist:.4f}\t'
                 'Val epoch_canberra_dist: {epoch_canberra_dist:.4f}\t'
                 'Val epoch_cosine_similarity: {epoch_cosine_similarity:.4f}\t'
                 'Val epoch_intersection_similarity: {epoch_intersection_similarity:.4f}\t'.format(epoch=epoch_num + 1,
                                                                                                   epoch_euclidean=ave_euclidean,
                                                                                                   epoch_chebyshev=ave_chebyshev,
                                                                                                   epoch_kldist=ave_kldist,
                                                                                                   epoch_clark_dist=ave_clark,
                                                                                                   epoch_canberra_dist=ave_canberra,
                                                                                                   epoch_cosine_similarity=ave_cosine,
                                                                                                   epoch_intersection_similarity=ave_intersection))

    # multi-label classification
    result.write("\n================================================================================\n")
    result.write('Epoch: {epoch:.1f}\t'
                 'Val epoch_eb_accuracy: {epoch_eb_accuracy:.4f}\t'
                 'Val epoch_eb_precision: {epoch_eb_precision:.4f}\t'
                 'Val epoch_eb_recall: {epoch_eb_recall:.4f}\t'
                 'Val epoch_eb_fbeta: {epoch_eb_fbeta:.4f}\t'
                 'Val epoch_hammingloss: {epoch_hammingloss:.4f}\t'
                 'Val epoch_oneError: {epoch_oneError:.4f}\t'
                 'Val epoch_coverage: {epoch_coverage:.4f}\t'
                 'Val epoch_Averageprecision: {epoch_Averageprecision:.4f}\t'
                 'Val epoch_rankingloss: {epoch_rankingloss:.4f}\t'
                 'Val epoch_accuracyMacro: {epoch_accuracyMacro:.4f}\t'
                 'Val epoch_accuracyMicro: {epoch_accuracyMicro:.4f}\t'
                 'Val epoch_precisionMacro: {epoch_precisionMacro:.4f}\t'
                 'Val epoch_precisionMicro: {epoch_precisionMicro:.4f}\t'
                 'Val epoch_recallMacro: {epoch_recallMacro:.4f}\t'
                 'Val epoch_recallMicro: {epoch_recallMicro:.4f}\t'
                 'Val epoch_fbetaMacro: {epoch_fbetaMacro:.4f}\t'
                 'Val epoch_fbetaMicro: {epoch_fbetaMicro:.4f}\t'.format(epoch=epoch_num + 1,
                                                                         epoch_eb_accuracy=ave_eb_accuracy,
                                                                         epoch_eb_precision=ave_eb_precision,
                                                                         epoch_eb_recall=ave_eb_recall,
                                                                         epoch_eb_fbeta=ave_eb_fbeta,
                                                                         epoch_hammingloss=ave_hammingLoss,
                                                                         epoch_oneError=ave_oneError,
                                                                         epoch_coverage=ave_coverage,
                                                                         epoch_Averageprecision=ave_averagePrecision,
                                                                         epoch_rankingloss=ave_rankingLoss,
                                                                         epoch_accuracyMacro=ave_accuracyMacro,
                                                                         epoch_accuracyMicro=ave_accuracyMicro,
                                                                         epoch_precisionMacro=ave_precisionMacro,
                                                                         epoch_precisionMicro=ave_precisionMicro,
                                                                         epoch_recallMacro=ave_recallMacro,
                                                                         epoch_recallMicro=ave_recallMicro,
                                                                         epoch_fbetaMacro=ave_fbetaMacro,
                                                                         epoch_fbetaMicro=ave_fbetaMicro))

    # Pearson correlation
    emopcc += pearsonr(dis_prediction.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
    # 每一个epoch loss平均
    epoch_loss = val_loss / len(valSet)
    # 每一个epoch pcc平均
    epoch_pcc = emopcc / len(valSet)
    epoch_accuracy = val_accuracy / cnt
    print('********', len(valSet))
    print('********', cnt)
    # validation loss
    val_loss = epoch_loss
    print("Validation: Epoch emotion distribution val_loss:", val_loss, "\nEpoch emotion distribution PCC:",
          epoch_pcc.item(), "\n", "==========================")
    result.write("\n------------------------------------------------------------------\n")
    result.write('Epoch: [{0}]\t' "Validation: Epoch emotion distribution val_loss: {val_loss: .4f}\t"
                 "\nEpoch emotion distribution PCC: {PCC: .4f}\t".format(epoch_num + 1, val_loss=val_loss,
                                                                         PCC=epoch_pcc))
    result.write('\n*****************************Epoch: [{0}]\t end************************\n'.format(epoch_num + 1))

    # checkpoint
    checkpoint = {
        'epoch': epoch_num + 1,
        'valid_loss_min_kl': epoch_loss,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path + "/train_co_attn_multi_dis_current_checkpoint.pt",
             best_model_path + "/train_co_attn_multi_dis_best_model.pt")

    ## TODO: save the model if validation loss has decreased
    # if epoch_cosine > epoch_cosine_min:
    #     print('Validation cosine creased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_cosine_min, epoch_cosine))
    #     # save checkpoint as best model
    #     save_ckp(checkpoint, True, checkpoint_path + "/train_co_attn_current_checkpoint.pt",
    #              best_model_path + "/train_co_attn_best_model.pt")
    #     epoch_cosine_min = epoch_cosine

    if epoch_accuracy > epoch_accuracy_min:
        print('Validation accuracy creased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_accuracy_min,
                                                                                          epoch_accuracy))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path + "/train_co_attn_multi_dis_current_checkpoint.pt",
                 best_model_path + "/train_co_attn_multi_dis_best_model.pt")
        epoch_accuracy_min = epoch_accuracy

    print('\n*****************************Epoch: [{0}]\t end************************\n'.format(epoch_num + 1))

final_eucliden = epoch_euclidean / num_epochs
final_chebyshev = epoch_chebyshev / num_epochs
final_kldist = epoch_kldist / num_epochs
final_clark = epoch_clark / num_epochs
final_canberra = epoch_canberra / num_epochs
final_cosine = epoch_cosine / num_epochs
final_intersection = epoch_intersection / num_epochs
final_hammingLoss = epoch_hammingLoss / num_epochs
final_eb_accuracy = epoch_eb_accuracy / num_epochs
final_eb_precision = epoch_eb_precision / num_epochs
final_eb_recall = epoch_eb_recall / num_epochs
final_eb_fbeta = epoch_eb_fbeta / num_epochs
final_oneError = epoch_oneError / num_epochs
final_averagePrecision = epoch_averagePrecision / num_epochs
final_rankingLoss = epoch_rankingLoss / num_epochs
final_accuracyMacro = epoch_accuracyMacro / num_epochs
final_fbetaMicro = epoch_fbetaMicro / num_epochs
final_coverage = epoch_coverage / num_epochs
final_accuracyMicro = epoch_accuracyMicro / num_epochs
final_precisionMacro = epoch_precisionMacro / num_epochs
final_precisionMicro = epoch_precisionMicro / num_epochs
final_recallMacro = epoch_recallMacro / num_epochs
final_recallMicro = epoch_recallMicro / num_epochs
final_fbetaMacro = epoch_fbetaMacro / num_epochs

result.write("\n================================================================================\n")
result.write('Epoch: {epoch:.1f}\t'
             'Val final_euclidean: {final_euclidean:.4f}\t'
             'Val final_chebyshev: {final_chebyshev:.4f}\t'
             'Val final_kldist: {final_kldist:.4f}\t'
             'Val final_clark_dist: {final_clark_dist:.4f}\t'
             'Val final_canberra_dist: {final_canberra_dist:.4f}\t'
             'Val final_cosine_similarity: {final_cosine_similarity:.4f}\t'
             'Val final_intersection_similarity: {final_intersection_similarity:.4f}\t'.format(epoch=epoch_num + 1,
                                                                                               final_euclidean=final_eucliden,
                                                                                               final_chebyshev=final_chebyshev,
                                                                                               final_kldist=final_kldist,
                                                                                               final_clark_dist=final_clark,
                                                                                               final_canberra_dist=final_canberra,
                                                                                               final_cosine_similarity=final_cosine,
                                                                                               final_intersection_similarity=final_intersection))

# multi-label classification
result.write("\n================================================================================\n")
result.write('Epoch: {epoch:.1f}\t'
             'Val final_eb_accuracy: {final_eb_accuracy:.4f}\t'
             'Val final_eb_precision: {final_eb_precision:.4f}\t'
             'Val final_eb_recall: {final_eb_recall:.4f}\t'
             'Val final_eb_fbeta: {final_eb_fbeta:.4f}\t'
             'Val final_hammingloss: {final_hammingloss:.4f}\t'
             'Val final_oneError: {final_oneError:.4f}\t'
             'Val final_coverage: {final_coverage:.4f}\t'
             'Val final_averageprecision: {final_averageprecision:.4f}\t'
             'Val final_rankingloss: {final_rankingloss:.4f}\t'
             'Val final_accuracyMacro: {final_accuracyMacro:.4f}\t'
             'Val final_accuracyMicro: {final_accuracyMicro:.4f}\t'
             'Val final_precisionMacro: {final_precisionMacro:.4f}\t'
             'Val final_precisionMicro: {final_precisionMicro:.4f}\t'
             'Val final_recallMacro: {final_recallMacro:.4f}\t'
             'Val final_recallMicro: {final_recallMicro:.4f}\t'
             'Val final_fbetaMacro: {final_fbetaMacro:.4f}\t'
             'Val final_fbetaMicro: {final_fbetaMicro:.4f}\t'.format(epoch=epoch_num + 1,
                                                                     final_eb_accuracy=final_eb_accuracy,
                                                                     final_eb_precision=final_eb_precision,
                                                                     final_eb_recall=final_eb_recall,
                                                                     final_eb_fbeta=final_eb_fbeta,
                                                                     final_hammingloss=final_hammingLoss,
                                                                     final_oneError=final_oneError,
                                                                     final_coverage=final_coverage,
                                                                     final_averageprecision=final_averagePrecision,
                                                                     final_rankingloss=final_rankingLoss,
                                                                     final_accuracyMacro=final_accuracyMacro,
                                                                     final_accuracyMicro=final_accuracyMicro,
                                                                     final_precisionMacro=final_precisionMacro,
                                                                     final_precisionMicro=final_precisionMicro,
                                                                     final_recallMacro=final_recallMacro,
                                                                     final_recallMicro=final_recallMicro,
                                                                     final_fbetaMacro=final_fbetaMacro,
                                                                     final_fbetaMicro=final_fbetaMicro))

# # testing
# net = EEGEncoder(args)
# net, optimizer, start_epoch, valid_loss_min_kl = load_ckp(
#     best_model_path + "/train_co_attn_multi_dis_best_model.pt", net, optimizer)
# net.eval()
# test_kl = 0
# emopcc = 0
# sum_euclidean = 0
# sum_chebyshev = 0
# sum_kldist = 0
# sum_clark = 0
# sum_canberra = 0
# sum_cosine = 0
# sum_intersection = 0
# sum_hammingLoss = 0
# sum_eb_accuracy = 0
# sum_eb_precision = 0
# sum_eb_recall = 0
# sum_eb_fbeta = 0
# sum_oneError = 0
# sum_averagePrecision = 0
# sum_rankingLoss = 0
# sum_accuracyMacro = 0
# sum_fbetaMicro = 0
# sum_converage = 0
# sum_accuracyMicro = 0
# sum_precisionMacro = 0
# sum_precisionMicro = 0
# sum_recallMacro = 0
# sum_recallMicro = 0
# sum_fbetaMacro = 0
# count = 0
# for i, data in enumerate(testDataloader):
#     count = count + 1
#     st_time = time.time()
#     test, dis, dom_label, left, right = data
#     dis = dis
#
#     if args['use_cuda']:
#         test = torch.nn.Parameter(test).cuda()
#         dis = torch.nn.Parameter(dis).cuda()
#         dom_label = torch.nn.Parameter(dom_label.float()).cuda()
#         left = torch.nn.Parameter(left).cuda()
#         right = torch.nn.Parameter(right).cuda()
#
#     # Forward pass
#     predict, att_1 = net(test, left, right, dis)
#
#     #for loss1
#     softmax = torch.nn.Softmax(dim=1)
#     dis_prediction = softmax(predict)
#     dis_prediction = dis_prediction.squeeze(dim=0)# [32,9]
#     dis = torch.squeeze(dis, dim=1)
#     loss1 = kl_div(dis_prediction.log(), dis)
#     #for loss2
#     predict_sig = torch.sigmoid(predict)
#     #loss2: BCELoss
#     dom_label = dom_label.detach()
#     loss2 = BCE(predict, dom_label)
#     #loss2: MLSML
#     # loss2 = MLSML(predict.cuda(), target_gt.cuda())
#     loss = lamda * loss1 + (1 - lamda) * loss2
#     test_loss = loss.item()
#     test_loss += test_loss / dis.shape[0]
#
#     # emotion distribution metrics
#     # euclidean
#     euclidean = euclidean_dist(dis.shape[0], dis, dis_prediction)
#     sum_euclidean += euclidean
#     # chebyshev
#     chebyshev = chebyshev_dist(dis.shape[0], dis, dis_prediction)
#     sum_chebyshev += chebyshev
#     # Kullback-Leibler divergence
#     kldist = KL_dist(dis, dis_prediction)
#     sum_kldist += kldist
#     # clark
#     clark = clark_dist(dis, dis_prediction)
#     sum_clark += clark
#     # canberra
#     canberra = canberra_dist(dis, dis_prediction)
#     sum_canberra += canberra
#     # cosine
#     cosine = cosine_dist(dis, dis_prediction)
#     sum_cosine += cosine
#     # intersection
#     intersection = intersection_dist(dis, dis_prediction)
#     sum_intersection += intersection
#
#     # for multilabel prediction
#     # example-based-classification
#     test_hammingLoss = hammingLoss(dom_label, predict_sig)
#     sum_hammingLoss += test_hammingLoss
#     test_eb_accuracy = accuracy(dom_label, predict_sig)
#     sum_eb_accuracy += test_eb_accuracy
#     test_eb_precision = precision(dom_label, predict_sig)
#     sum_eb_precision += test_eb_precision
#     test_eb_recall = recall(dom_label, predict_sig)
#     sum_eb_recall += test_eb_recall
#     test_eb_fbeta = fbeta(dom_label, predict_sig)
#     sum_eb_fbeta += test_eb_fbeta
#
#     # example-based-ranking
#     test_oneError = oneError(dom_label, predict_sig)
#     sum_oneError += test_oneError
#     test_coverage = coverage(dom_label, predict_sig)
#     sum_coverage += test_coverage
#     test_averagePrecision = averagePrecision(dom_label, predict_sig)
#     sum_averagePrecision += test_averagePrecision
#     test_rankingLoss = rankingLoss(dom_label, predict_sig)
#     sum_rankingLoss += test_rankingLoss
#
#     # label-based-classification
#     test_accuracyMacro = accuracyMacro(dom_label, predict_sig)
#     sum_accuracyMacro += test_accuracyMacro
#     test_accuracyMicro = accuracyMicro(dom_label, predict_sig)
#     sum_accuracyMicro += test_accuracyMicro
#     test_precisionMacro = precisionMacro(dom_label, predict_sig)
#     sum_precisionMacro += test_precisionMacro
#     test_precisionMicro = precisionMicro(dom_label, predict_sig)
#     sum_precisionMicro += test_precisionMicro
#     test_recallMacro = recallMacro(dom_label, predict_sig)
#     sum_recallMacro += test_recallMacro
#     test_recallMicro = recallMicro(dom_label, predict_sig)
#     sum_precisionMicro += test_precisionMicro
#     test_fbetaMacro = fbetaMacro(dom_label, predict_sig)
#     sum_fbetaMacro += test_fbetaMacro
#     test_fbetaMicro = fbetaMicro(dom_label, predict_sig)
#     sum_fbetaMicro += test_fbetaMicro
#
#
#     if i % 10 == 0:
#         print('euclidean_dist: {euclidean_dist:.4f}\t'
#                   'chebyshev_dist: {chebyshev_dist:.4f}\t'
#                   'kldist: {kldist:.4f}\t'
#                   'clark_dist: {clark_dist:.4f}\t'
#                   'canberra_dist: {canberra_dist:.4f}\t'
#                   'cosine_dist: {cosine_dist:.4f}\t'
#                   'intersection_dist: {intersection_dist:.4f}\n'.format(euclidean_dist=euclidean,
#                                                                         chebyshev_dist=chebyshev, kldist=kldist,
#                                                                         clark_dist=clark, canberra_dist=canberra,
#                                                                         cosine_dist=cosine,
#                                                                         intersection_dist=intersection))
#
#         print("Test set results:\n",
#               'Multilabel metrics: example-based-classification:\n',
#               "hammingLoss= {:.4f}".format(test_hammingLoss),
#               "accuracy= {:.4f}".format(test_eb_accuracy),
#               "precision= {:.4f}".format(test_eb_precision),
#               "recall= {:.4f}".format(test_eb_recall),
#               "fbeta= {:.4f}".format(test_eb_fbeta))
#
#         print("Test set results:\n",
#               'Multilabel metrics: example-based-ranking:\n',
#               "oneError= {:.4f}".format(test_oneError),
#               "coverage= {:.4f}".format(test_coverage),
#               "averagePrecision= {:.4f}".format(test_averagePrecision),
#               "rankingLoss= {:.4f}".format(test_rankingLoss))
#
#         print("Test set results:\n",
#               'Multilabel metrics: label-based-classification:\n',
#               "accuracyMacro= {:.4f}".format(test_accuracyMacro),
#               "accuracyMicro= {:.4f}".format(test_accuracyMicro),
#               "precisionMacro= {:.4f}".format(test_precisionMacro),
#               "precisionMicro= {:.4f}".format(test_precisionMicro),
#               "recallMacro= {:.4f}".format(test_recallMacro),
#               "recallMicro= {:.4f}".format(test_recallMicro),
#               "fbetaMacro= {:.4f}".format(test_fbetaMacro),
#               "fbetaMicro= {:.4f}".format(test_fbetaMicro))
#
#
#         result.write("\n------------------------------------------------------------------\n")
#         result.write('Test co-attention:\n ')
#         result.write('att_1: \t')
#         result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
#
#         result.write('\n Epoch: [{0}]: Test....\n'.format(epoch_num+1))
#         result.write("\n========================================\n")
#         result.write('euclidean_dist: {euclidean_dist:.4f}\t'
#                   'chebyshev_dist: {chebyshev_dist:.4f}\t'
#                   'kldist: {kldist:.4f}\t'
#                   'clark_dist: {clark_dist:.4f}\t'
#                   'canberra_dist: {canberra_dist:.4f}\t'
#                   'cosine_dist: {cosine_dist:.4f}\t'
#                   'intersection_dist: {intersection_dist:.4f}\t'.format(euclidean_dist=euclidean,
#                                                                         chebyshev_dist=chebyshev, kldist=kldist,
#                                                                         clark_dist=clark, canberra_dist=canberra,
#                                                                         cosine_dist=cosine,
#                                                                         intersection_dist=intersection))
#         result.write("\n------------------------------------------------------------------\n")
#         result.write("Test set results:\n")
#         result.write('Multilabel metrics: example-based-classification:\n')
#         result.write(
#                      'hammingLoss: {hammingLoss:.4f}\t'
#                      'accuracy: {accuracy:.4f}\t'
#                      'precision: {precision:.4f}\t'
#                      'recall: {recall:.4f}\t'
#                      'fbeta: {fbeta:.4f}\t'.format(
#                                                    hammingLoss=test_hammingLoss,
#                                                    accuracy=test_eb_accuracy, precision=test_eb_precision,
#                                                    recall=test_eb_recall, fbeta=test_eb_fbeta))
#         result.write("\n")
#         result.write('Multilabel metrics: example-based-ranking:\n')
#         result.write('oneError: {oneError:.4f}\t'
#                      'coverage: {coverage:.4f}\t'
#                      'averagePrecision: {averagePrecision:.4f}\t'
#                      'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=test_oneError,
#                                                                coverage=test_coverage,
#                                                                averagePrecision=test_averagePrecision,
#                                                                rankingLoss=test_rankingLoss))
#         result.write("\n")
#         result.write('Multilabel metrics: label-based-classification:\n')
#         result.write('accuracyMacro: {accuracyMacro:.4f}\t'
#                      'accuracyMicro: {accuracyMicro:.4f}\t'
#                      'precisionMacro: {precisionMacro:.4f}\t'
#                      'precisionMicro: {precisionMicro:.4f}\t'
#                      'recallMacro: {recallMacro:.4f}\t'
#                      'recallMicro: {recallMicro:.4f}\t'
#                      'fbetaMacro: {fbetaMacro:.4f}\t'
#                      'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=test_accuracyMacro,
#                                                              accuracyMicro=test_accuracyMicro,
#                                                              precisionMacro=test_precisionMacro,
#                                                              precisionMicro=test_precisionMicro,
#                                                              recallMacro=test_recallMacro,
#                                                              recallMicro=test_recallMicro,
#                                                              fbetaMacro=test_fbetaMacro,
#                                                              fbetaMicro=test_fbetaMicro))
#     # pearson correlation
#     emopcc += pearsonr(dis_prediction.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
# # average loss
# test_testkl = test_loss / len(testSet)
# # average pcc
# test_emopcc = emopcc / len(testSet)
# ave_euclidean = sum_euclidean/count
# ave_chebyshev = sum_chebyshev/count
# ave_kldist = sum_kldist/count
# ave_clark = sum_clark/count
# ave_canberra = sum_canberra/count
# ave_cosine = sum_cosine/count
# ave_intersection = sum_intersection/count
# ave_hammingLoss = sum_hammingLoss/count
# ave_eb_accuracy = sum_eb_accuracy/count
# ave_eb_precision = sum_eb_precision/count
# ave_eb_recall = sum_eb_recall/count
# ave_eb_fbeta = sum_eb_fbeta/count
# ave_oneError = sum_oneError/count
# ave_averagePrecision = sum_averagePrecision/count
# ave_rankingLoss = sum_rankingLoss/count
# ave_accuracyMacro = sum_accuracyMacro/count
# ave_fbetaMicro = sum_fbetaMicro/count
# ave_coverage = sum_coverage / count
# ave_accuracyMicro = sum_accuracyMicro / count
# ave_precisionMacro = sum_precisionMacro / count
# ave_precisionMicro = sum_precisionMicro / count
# ave_recallMacro = sum_recallMacro / count
# ave_recallMicro = sum_recallMicro / count
#
# print("\n================================================================================\n")
# print("Test Emotion distribution test_loss:", test_testkl, "\Test Emotion distribution PCC:", test_emopcc.item())
# result.write("\n------------------------------------------------------------------\n")
# result.write('Epoch: [{0}]\t' "Test Emotion distribution test_loss:{test_loss: .4f}\n"
#              "Test Emotion distribution PCC:{PCC: .4f}\t".format(epoch_num+1, test_loss=test_testkl,PCC=test_emopcc))
#
# result.write("\n================================================================================\n")
# result.write('Epoch: {epoch:.1f}\t'
#                      'Test epoch_euclidean: {epoch_euclidean:.4f}\t'
#                      'Test epoch_chebyshev: {epoch_chebyshev:.4f}\t'
#                      'Test epoch_kldist: {epoch_kldist:.4f}\t'
#                      'Test epoch_clark_dist: {epoch_clark_dist:.4f}\t'
#                      'Test epoch_canberra_dist: {epoch_canberra_dist:.4f}\t'
#                      'Test epoch_cosine_similarity: {epoch_cosine_similarity:.4f}\t'
#                      'Test epoch_intersection_similarity: {epoch_intersection_similarity:.4f}\t'.format(epoch=epoch_num+1,
#                                                                                                         epoch_euclidean=ave_euclidean,
#                                                                                                         epoch_chebyshev=ave_chebyshev,
#                                                                                                         epoch_kldist=ave_kldist,
#                                                                                                         epoch_clark_dist=ave_clark,
#                                                                                                         epoch_canberra_dist=ave_canberra,
#                                                                                                         epoch_cosine_similarity=ave_cosine,
#                                                                                                         epoch_intersection_similarity=ave_intersection))
#
# #multi-label classification
# result.write("\n================================================================================\n")
# result.write('Epoch: {epoch:.1f}\t'
#                      'Test epoch_accuracy: {epoch_accuracy:.4f}\t'
#                      'Test epoch_eb_accuracy: {epoch_eb_accuracy:.4f}\t'
#                      'Test epoch_eb_precision: {epoch_eb_precision:.4f}\t'
#                      'Test epoch_eb_recall: {epoch_eb_recall:.4f}\t'
#                      'Test epoch_eb_fbeta: {epoch_eb_fbeta:.4f}\t'
#                      'Test epoch_hammingloss: {epoch_hammingloss:.4f}\t'
#                      'Test epoch_oneError: {epoch_oneError:.4f}\t'
#                      'Test epoch_coverage: {epoch_coverage:.4f}\t'
#                      'Test epoch_Averageprecision: {epoch_Averageprecision:.4f}\t'
#                      'Test epoch_rankingloss: {epoch_rankingloss:.4f}\t'
#                      'Test epoch_accuracyMacro: {epoch_accuracyMacro:.4f}\t'
#                      'Test epoch_accuracyMicro: {epoch_accuracyMicro:.4f}\t'
#                      'Test epoch_precisionMacro: {epoch_precisionMacro:4f}\t'
#                      'Test epoch_precisionMicro: {epoch_precisionMicro:.4f}\t'
#                      'Test epoch_recallMacro: {epoch_recallMacro:.4f}\t'
#                      'Test epoch_recallMicro: {epoch_recallMicro:.4f}\t'
#                      'Test epoch_fbetaMacro: {epoch_fbetaMacro:.4f}\t'
#                      'Test epoch_fbetaMicro: {epoch_fbetaMicro:.4f}\t'.format(epoch=epoch_num+1,
#                                                                             epoch_accuracy=epoch_accuracy,
#                                                                             epoch_eb_accuracy=ave_eb_accuracy,
#                                                                             epoch_eb_precision=ave_eb_precision,
#                                                                             epoch_eb_recall=ave_eb_recall,
#                                                                             epoch_eb_fbeta=ave_eb_fbeta,
#                                                                             epoch_hammingloss=ave_hammingLoss,
#                                                                             epoch_oneError=ave_oneError,
#                                                                             epoch_coverage=ave_coverage,
#                                                                             epoch_Averageprecision=ave_averagePrecision,
#                                                                             epoch_rankingloss=ave_rankingLoss,
#                                                                             epoch_accuracyMacro=ave_accuracyMacro,
#                                                                             epoch_accuracyMicro=ave_accuracyMicro,
#                                                                             epoch_precisionMacro=ave_precisionMacro,
#                                                                             epoch_precisionMicro=ave_precisionMicro,
#                                                                             epoch_recallMacro=ave_recallMacro,
#                                                                             epoch_recallMicro=ave_recallMicro,
#                                                                             epoch_fbetaMacro=ave_fbetaMacro,
#                                                                             epoch_fbetaMicro=ave_fbetaMicro))


print(time_att.cpu().detach().numpy().mean(axis=0))
print(channel_att.cpu().detach().numpy().mean(axis=0))

# import csv
#
# with open("GC_POSITIVE.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(GC_est)
