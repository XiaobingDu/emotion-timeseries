# -*-coding:utf-8-*-

from __future__ import print_function
from model_co_attn import MovieNet
from dataManager import five_fold, dataSplit, get_sample_data, get_sample_data_withoutOverlap
from utils_co_attn import *
from torch.utils.data import DataLoader
from scipy.stats.mstats import pearsonr
from clstm import train_model_gista
import time, argparse
import codecs
from multilabelMetrics.examplebasedclassification import *
from multilabelMetrics.examplebasedranking import *
from multilabelMetrics.labelbasedclassification import *
from multilabelMetrics.labelbasedranking import *
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--path1', type=str, choices=["/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_addLabel_sum1/",
                                                  '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_9_DOM/',
                                                  '../EEG_PSD_9_DOM/'])
parser.add_argument('--path2', type=str, choices=["/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/featureAll.mat",
                                                  '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/DOM_featureAll.mat',
                                                  '../DOM_feature_all/DOM_featureAll.mat'])
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--iter_num', type=int,
                    help='Number of iterate to train.')
parser.add_argument('--lamda', type=float, default=0.6,
                    help='The lamda is the weight to control the trade-off between two type losses.')
parser.add_argument('--overlap', type=str, default='with', choices=['with','without'],
                    help='Get the samples with/without time overlap.')
parser.add_argument('--sub_id', type=int, default=0,
                    help='The subject ID for Test.')
parser.add_argument('--fold_id', type=int, default= 1,
                    help='The fold id  for Test.')
parser.add_argument('--epochs', type=int, default = 20,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--label_num', type=int, default=9)
parser.add_argument('--db_name', type=str, default='LDL_data')
parser.add_argument('--strategy', type=str, default='five_fold', choices=['split','five_fold','ldl_loso'])
parser.add_argument('--save_file', type=str, default= 'co_attn_GC')
parser.add_argument('--log_dir', type=str)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--bidirectional', type=bool, default=True )
parser.add_argument('--GCN_hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--LSTM_layers', type=int, default=2,
                    help='Number of LSTM layers.')
parser.add_argument('--LSTM_hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--attn_len', type=int, default=10,
                    help='attn_len = time_sequence')
parser.add_argument('--out_layer', type=int, default=9)
parser.add_argument('--encoder_size', type=int, default=64)
parser.add_argument('--decoder_size', type=int, default=128)
parser.add_argument('--dyn_embedding_size', type=int, default=32)
parser.add_argument('--input_embedding_size', type=int, default=32)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
FLAGS = parser.parse_args()

args = {}
# path1 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_9_DOM/'
# path2 = '/home/xiaobingdu/EEG_experiments/LDL-LSTM_softmax/attn_lstm/EEG_PSD_multilabel_9_win/DOM_featureAll.mat'
path1 = FLAGS.path1
path2 = FLAGS.path2
db_name = 'LDL_data'
best_model_path ="./best_model"
checkpoint_path ="./checkpoints"
valid_loss_min =np.Inf
epoch_cosine_min = np.NINF
epoch_accuracy_min = np.NINF

## Network Arguments
args['Frontal_len'] = 35
args['Temporal_len'] = 40
args['Central_len'] = 45
args['Parietal_len'] = 15
args['Occipital_len'] = 15
args['use_cuda'] = True
args['train_flag'] = True
args['optimizer'] = 'adam'
args['model_path'] = 'trained_models/EEG_eval_model.tar'
args['out_layer'] = FLAGS.out_layer #64 #9 #2048 same as GCN output_size
args['dropout_prob'] = FLAGS.dropout
args['encoder_size'] = FLAGS.encoder_size
args['decoder_size'] = FLAGS.decoder_size
args['dyn_embedding_size'] = FLAGS.dyn_embedding_size
args['input_embedding_size'] = FLAGS.input_embedding_size
args['embed_dim'] = FLAGS.embed_dim
args['h_dim'] =  FLAGS.LSTM_hidden #32 #512
args['n_layers'] =FLAGS.LSTM_layers
args['attn_len'] = FLAGS.attn_len
num_epochs = FLAGS.epochs
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
lamda = FLAGS.lamda
overlap = FLAGS.overlap
strategy = FLAGS.strategy
fold_id = FLAGS.fold_id
GC_est =None
# 对应5个脑区的电极idx：Frontal、Temporal、Central、Parietal、Occipital
idx = [[0 ,1 ,2 ,3 ,4 ,5 ,6] ,[7 ,11 ,12 ,16 ,17 ,21 ,22 ,26] ,[8 ,9 ,10 ,13 ,14 ,15 ,18 ,19 ,20] ,[23 ,24 ,25]
       ,[27 ,28 ,29]]

# load train, val, test data
if overlap == 'with':
    print('------overlap: True------')
    data_set = get_sample_data(path1,path2)
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
    train_data, val_data, test_data, train_dis, val_dis, test_dis, train_dom_label, val_dom_label, test_dom_label = dataSplit(path1, data_set, db_name)

train_data = train_data.astype(np.float32)
val_data = val_data.astype(np.float32)
test_data = test_data.astype(np.float32)

train_dis = train_dis.astype(np.float32)
val_dis = val_dis.astype(np.float32)
test_dis = test_dis.astype(np.float32)

train_dom_label = train_dom_label.astype(np.float32)
val_dom_label = val_dom_label.astype(np.float32)
test_dom_label = test_dom_label.astype(np.float32)

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
    (net.parameters() ,lr=lr, weight_decay=1e-8) #weight_decay=0.9
# mse = torch.nn.MSELoss(reduction='sum')
kl_div = torch.nn.KLDivLoss(size_average = True, reduce = True)
#from multi-label dom_emotion predict
MLSML = torch.nn.MultiLabelSoftMarginLoss()
mutilabel_criterion = torch.nn.BCELoss()

# write in file and save
result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
if FLAGS.strategy=="ldl_loso":
        result.write('========\nsubject %s\n========\n' % str(FLAGS.sub_id))
elif FLAGS.strategy=="five_fold":
        result.write('\n========\nfold %s\n========\n' % str(FLAGS.fold_id))
        result.write('\n========\nlamda %s\n========\n' % str(FLAGS.lamda))
elif FLAGS.strategy == "split":
    result.write('\n========\nsplit %s\n========\n' % '5:3:2')
    result.write('\n========\nlamda %s\n========\n' % str(FLAGS.lamda))
result.write('model parameters: %s' % str(FLAGS))
result.close()


#traning&val&test
for epoch_num in range(num_epochs):
    #    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_learning_rate(optimizer, epoch_num, lr)


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

        # Forward pass
        predict, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 \
            = net(train, Frontal, Temporal, Central, Parietal, Occipital, dis)
        #get GC
        train_model_gista(shared_encoder, input_clstm, lam=0.5, lam_ridge=1e-4, lr=0.001, max_iter=1, check_every=1000, truncation=64)
        GC_est = shared_encoder.GC().cpu().data.numpy()

        # for loss1
        # softmax layer
        softmax = torch.nn.Softmax(dim=1)
        dis_prediction = softmax(predict)
        dis_prediction = dis_prediction.squeeze(dim=0)  # [32,9]
        dis = torch.squeeze(dis, dim=1)
        # loss1: KLDivLoss
        loss1 = kl_div(dis_prediction.log(), dis)
        # for loss2
        label_prediction = torch.sigmoid(predict)
        # loss2: BCELoss
        target_gt = dom_label
        target_gt = target_gt.detach()
        loss2 = mutilabel_criterion(label_prediction, target_gt)
        # print('*' * 100)
        # print('predict label...', label_prediction)
        # loss2: MLSML
        # loss2 = MLSML(label_prediction.cuda(), target_gt.cuda())
        loss = lamda * loss1 + (1 - lamda) * loss2

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
        train_subsetAccuracy = subsetAccuracy(dom_label, label_prediction)
        train_hammingLoss = hammingLoss(dom_label, label_prediction)
        train_eb_accuracy = accuracy(dom_label, label_prediction)
        train_eb_precision = precision(dom_label, label_prediction)
        train_eb_recall = recall(dom_label, label_prediction)
        train_eb_fbeta = fbeta(dom_label, label_prediction)
        #
        # # example-based-ranking
        train_oneError = oneError(dom_label, label_prediction)
        # train_coverage = coverage(dom_label, label_prediction)
        train_averagePrecision = averagePrecision(dom_label, label_prediction)
        train_rankingLoss = rankingLoss(dom_label, label_prediction)
        #
        # # label-based-classification
        train_accuracyMacro = accuracyMacro(dom_label, label_prediction)
        # train_accuracyMicro = accuracyMicro(dom_label, label_prediction)
        # train_precisionMacro = precisionMacro(dom_label, label_prediction)
        # train_precisionMicro = precisionMicro(dom_label, label_prediction)
        # train_recallMacro = recallMacro(dom_label, label_prediction)
        # train_recallMicro = recallMicro(dom_label, label_prediction)
        # train_fbetaMacro = fbetaMacro(dom_label, label_prediction)
        train_fbetaMicro = fbetaMicro(dom_label, label_prediction)

        # # label-based-ranking
        # train_aucMacro = aucMacro(dom_label, label_prediction)
        # train_aucMicro = aucMicro(dom_label, label_prediction)
        # train_aucInstance = aucInstance(dom_label, label_prediction)

        #results
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
                  "fbeta= {:.4f}".format(train_eb_fbeta)
                  )

            print("Training set results:\n",
                  'Multilabel metrics: example-based-ranking:\n',
                  "oneError= {:.4f}".format(train_oneError),
                  "averagePrecision= {:.4f}".format(train_averagePrecision),
                  "rankingLoss= {:.4f}".format(train_rankingLoss))

            print("Training set results:\n",
                  'Multilabel metrics: label-based-classification:\n',
                  "accuracyMacro= {:.4f}".format(train_accuracyMacro),
                  "fbetaMicro= {:.4f}".format(train_fbetaMicro))

            # print("Training set results:\n",
            #       'Multilabel metrics: example-based-ranking:\n',
            #       "oneError= {:.4f}".format(train_oneError),
            #       "coverage= {:.4f}".format(train_coverage),
            #       "averagePrecision= {:.4f}".format(train_averagePrecision),
            #       "rankingLoss= {:.4f}".format(train_rankingLoss))
            #
            # print("Training set results:\n",
            #       'Multilabel metrics: label-based-classification:\n',
            #       "accuracyMacro= {:.4f}".format(train_accuracyMacro),
            #       "accuracyMicro= {:.4f}".format(train_accuracyMicro),
            #       "precisionMacro= {:.4f}".format(train_precisionMacro),
            #       "precisionMicro= {:.4f}".format(train_precisionMicro),
            #       "recallMacro= {:.4f}".format(train_recallMacro),
            #       "recallMicro= {:.4f}".format(train_recallMicro),
            #       "fbetaMacro= {:.4f}".format(train_fbetaMacro),
            #       "fbetaMicro= {:.4f}".format(train_fbetaMicro))
            #
            # print("Training set results:\n",
            #       'Multilabel metrics: label-based-ranking:\n',
            #       "aucMacro= {:.4f}".format(train_aucMacro),
            #       "aucMicro= {:.4f}".format(train_aucMicro),
            #       "aucInstance={:.4f}".format(train_aucInstance))

            result = codecs.open(FLAGS.save_file, 'a', 'utf-8')
            result.write("\n------------------------------------------------------------------\n")
            result.write("Training GC_est:\n")
            result.write('%s\n' % GC_est)
            result.write('Training co-attention:\n ')
            result.write('att_1: \t')
            result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
            result.write('att_2: \t')
            result.write('%s\n' % att_2.cpu().detach().numpy().mean(axis=0))
            result.write('att_3: \t')
            result.write('%s\n' % att_3.cpu().detach().numpy().mean(axis=0))
            result.write('att_4: \t')
            result.write('%s\n' % att_4.cpu().detach().numpy().mean(axis=0))
            result.write('att_5: \t')
            result.write('%s\n' % att_5.cpu().detach().numpy().mean(axis=0))
            result.write('att_6: \t')
            result.write('%s\n' % att_6.cpu().detach().numpy().mean(axis=0))
            result.write('att_7: \t')
            result.write('%s\n' % att_7.cpu().detach().numpy().mean(axis=0))
            result.write('att_8: \t')
            result.write('%s\n' % att_8.cpu().detach().numpy().mean(axis=0))
            result.write('att_9: \t')
            result.write('%s\n' % att_9.cpu().detach().numpy().mean(axis=0))
            result.write('att_10: \t')
            result.write('%s\n' % att_10.cpu().detach().numpy().mean(axis=0))

            result.write('\n Epoch: [{0}]: Training....\n'.format(epoch_num + 1))
            result.write("\n========================================\n")
            result.write('euclidean_dist: {euclidean_dist:.4f}\t'
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
            result.write("\n------------------------------------------------------------------\n")
            result.write("Training set results:\n")
            result.write('Multilabel metrics: example-based-classification:\n')
            result.write(
                         'subsetAccuracy: {subsetAccuracy:.4f}\t'
                         'hammingLoss: {hammingLoss:.4f}\t'
                         'accuracy: {accuracy:.4f}\t'
                         'precision: {precision:.4f}\t'
                         'recall: {recall:.4f}\t'
                         'fbeta: {fbeta:.4f}\t'.format(
                                                       subsetAccuracy=train_subsetAccuracy,
                                                       hammingLoss=train_hammingLoss,
                                                       accuracy=train_eb_accuracy,
                                                        precision=train_eb_precision,
                                                       recall=train_eb_recall, fbeta=train_eb_fbeta)
                                                         )

            result.write("\n")
            result.write('Multilabel metrics: example-based-ranking:\n')
            result.write('oneError: {oneError:.4f}\t'
                         'averagePrecision: {averagePrecision:.4f}\t'
                         'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=train_oneError,
                                                                   averagePrecision=train_averagePrecision,
                                                                   rankingLoss=train_rankingLoss))

            result.write("\n")
            result.write('Multilabel metrics: label-based-classification:\n')
            result.write('accuracyMacro: {accuracyMacro:.4f}\t'
                         'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=train_accuracyMacro,
                                                                 fbetaMicro=train_fbetaMicro))

            # result.write("\n")
            # result.write('Multilabel metrics: example-based-ranking:\n')
            # result.write('oneError: {oneError:.4f}\t'
            #              'coverage: {coverage:.4f}\t'
            #              'averagePrecision: {averagePrecision:.4f}\t'
            #              'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=train_oneError,
            #                                                        coverage=train_coverage,
            #                                                        averagePrecision=train_averagePrecision,
            #                                                        rankingLoss=train_rankingLoss))

            # result.write("\n")
            # result.write('Multilabel metrics: label-based-classification:\n')
            # result.write('accuracyMacro: {accuracyMacro:.4f}\t'
            #              'accuracyMicro: {accuracyMicro:.4f}\t'
            #              'precisionMacro: {precisionMacro:.4f}\t'
            #              'precisionMicro: {precisionMicro:.4f}\t'
            #              'recallMacro: {recallMacro:.4f}\t'
            #              'recallMicro: {recallMicro:.4f}\t'
            #              'fbetaMacro: {fbetaMacro:.4f}\t'
            #              'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=train_accuracyMacro,
            #                                                      accuracyMicro=train_accuracyMicro,
            #                                                      precisionMacro=train_precisionMacro,
            #                                                      precisionMicro=train_precisionMicro,
            #                                                      recallMacro=train_recallMacro,
            #                                                      recallMicro=train_recallMicro,
            #                                                      fbetaMacro=train_fbetaMacro,
            #                                                      fbetaMicro=train_fbetaMicro))
            #
            # result.write("\n")
            # result.write('Multilabel metrics: label-based-ranking:\n')
            # result.write('aucMacro: {aucMacro:.4f}\t'
            #              'aucMicro: {aucMicro:.4f}\t'
            #              'aucInstance: {aucInstance:.4f}\t'.format(aucMacro=train_aucMacro,
            #                                                        aucMicro=train_aucMicro,
            #                                                        aucInstance=train_aucInstance))

    print("Epoch no:", epoch_num + 1, "| Avg_train_loss:".format(avg_tr_loss / len(trSet), '0.4f'))
    result.write("\n------------------------------------------------------------------\n")
    result.write("Epoch no: {epoch: .4f}\t"
                 "| Avg_train_loss: {loss:.4f}\t".format(epoch=epoch_num + 1, loss=avg_tr_loss / len(trSet)))

    ## Validate:
    net.eval()
    val_kl = 0
    emopcc = 0
    val_cosine = 0
    val_accuracy = 0
    cnt = 0

    for i, data in enumerate(valDataloader):
        cnt += 1
        print("Val ..... 第 {} 个Batch.....".format(i))
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

        # Forward pass
        predict, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
            net(val, Frontal, Temporal, Central, Parietal, Occipital, dis)

        # for loss1
        # softmax layer
        softmax = torch.nn.Softmax(dim=1)
        dis_prediction = softmax(predict)
        dis_prediction = dis_prediction.squeeze(dim=0)  # [32,9]
        dis = torch.squeeze(dis, dim=1)
        # loss1: KLDivLoss
        loss1 = kl_div(dis_prediction.log(), dis)
        # for loss2
        label_prediction = torch.sigmoid(predict)
        # loss2: BCELoss
        dom_label = dom_label.detach()
        loss2 = mutilabel_criterion(label_prediction, dom_label)
        # print('*' * 100)
        # print('predict label...', label_prediction)
        # loss2: MLSML
        # loss2 = MLSML(label_prediction.cuda(), target_gt.cuda())
        loss = lamda * loss1 + (1 - lamda) * loss2

        val_loss = loss.item()
        val_loss += val_loss /dis.shape[0]

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
        val_cosine += cosine
        # intersection
        intersection = intersection_dist(dis, dis_prediction)

        # # for multilabel prediction
        # # example-based-classification
        val_hammingLoss = hammingLoss(dom_label, label_prediction)
        val_eb_accuracy = accuracy(dom_label, label_prediction)
        val_accuracy += val_eb_accuracy
        val_eb_precision = precision(dom_label, label_prediction)
        val_eb_recall = recall(dom_label, label_prediction)
        val_eb_fbeta = fbeta(dom_label, label_prediction)

        # example-based-ranking
        val_oneError = oneError(dom_label, label_prediction)
        val_averagePrecision = averagePrecision(dom_label, label_prediction)
        val_rankingLoss = rankingLoss(dom_label, label_prediction)

        # label-based-classification
        val_accuracyMacro = accuracyMacro(dom_label, label_prediction)
        val_fbetaMicro = fbetaMicro(dom_label, label_prediction)

        # # example-based-ranking
        # val_oneError = oneError(dom_label, label_prediction)
        # val_coverage = coverage(dom_label, label_prediction)
        # val_averagePrecision = averagePrecision(dom_label, label_prediction)
        # val_rankingLoss = rankingLoss(dom_label, label_prediction)
        #
        # # label-based-classification
        # val_accuracyMacro = accuracyMacro(dom_label, label_prediction)
        # val_accuracyMicro = accuracyMicro(dom_label, label_prediction)
        # val_precisionMacro = precisionMacro(dom_label, label_prediction)
        # val_precisionMicro = precisionMicro(dom_label, label_prediction)
        # val_recallMacro = recallMacro(dom_label, label_prediction)
        # val_recallMicro = recallMicro(dom_label, label_prediction)
        # val_fbetaMacro = fbetaMacro(dom_label, label_prediction)
        # val_fbetaMicro = fbetaMicro(dom_label, label_prediction)
        #
        # # label-based-ranking
        # val_aucMacro = aucMacro(dom_label, label_prediction)
        # val_aucMicro = aucMicro(dom_label, label_prediction)
        # val_aucInstance = aucInstance(dom_label, label_prediction)

        #results
        if i % 10 == 0:

            print('Validation euclidean_dist: {euclidean_dist:.4f}\t'
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

            print("Validation set results:\n",
                  'Multilabel metrics: example-based-classification:\n',
                  "hammingLoss= {:.4f}".format(val_hammingLoss),
                  "accuracy= {:.4f}".format(val_eb_accuracy),
                  "precision= {:.4f}".format(val_eb_precision),
                  "recall= {:.4f}".format(val_eb_recall),
                  "fbeta= {:.4f}".format(val_eb_fbeta))

            print(
                  'Multilabel metrics: example-based-ranking:\n',
                  "oneError= {:.4f}".format(val_oneError),
                  "averagePrecision= {:.4f}".format(val_averagePrecision),
                  "rankingLoss= {:.4f}".format(val_rankingLoss))

            print(
                  'Multilabel metrics: label-based-classification:\n',
                  "accuracyMacro= {:.4f}".format(val_accuracyMacro),
                  "fbetaMicro= {:.4f}".format(val_fbetaMicro))

            # print(
            #       'Multilabel metrics: example-based-ranking:\n',
            #       "oneError= {:.4f}".format(val_oneError),
            #       "coverage= {:.4f}".format(val_coverage),
            #       "averagePrecision= {:.4f}".format(val_averagePrecision),
            #       "rankingLoss= {:.4f}".format(val_rankingLoss))
            #
            # print(
            #       'Multilabel metrics: label-based-classification:\n',
            #       "accuracyMacro= {:.4f}".format(val_accuracyMacro),
            #       "accuracyMicro= {:.4f}".format(val_accuracyMicro),
            #       "precisionMacro= {:.4f}".format(val_precisionMacro),
            #       "precisionMicro= {:.4f}".format(val_precisionMicro),
            #       "recallMacro= {:.4f}".format(val_recallMacro),
            #       "recallMicro= {:.4f}".format(val_recallMicro),
            #       "fbetaMacro= {:.4f}".format(val_fbetaMacro),
            #       "fbetaMicro= {:.4f}".format(val_fbetaMicro))
            #
            # print(
            #       'Multilabel metrics: label-based-ranking:\n',
            #       "aucMacro= {:.4f}".format(val_aucMacro),
            #       "aucMicro= {:.4f}".format(val_aucMicro),
            #       "aucInstance={:.4f}".format(val_aucInstance))

            result.write("\n------------------------------------------------------------------\n")
            result.write('Validation co-attention:\n ')
            result.write('att_1: \t')
            result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
            result.write('att_2: \t')
            result.write('%s\n' % att_2.cpu().detach().numpy().mean(axis=0))
            result.write('att_3: \t')
            result.write('%s\n' % att_3.cpu().detach().numpy().mean(axis=0))
            result.write('att_4: \t')
            result.write('%s\n' % att_4.cpu().detach().numpy().mean(axis=0))
            result.write('att_5: \t')
            result.write('%s\n' % att_5.cpu().detach().numpy().mean(axis=0))
            result.write('att_6: \t')
            result.write('%s\n' % att_6.cpu().detach().numpy().mean(axis=0))
            result.write('att_7: \t')
            result.write('%s\n' % att_7.cpu().detach().numpy().mean(axis=0))
            result.write('att_8: \t')
            result.write('%s\n' % att_8.cpu().detach().numpy().mean(axis=0))
            result.write('att_9: \t')
            result.write('%s\n' % att_9.cpu().detach().numpy().mean(axis=0))
            result.write('att_10: \t')
            result.write('%s\n' % att_10.cpu().detach().numpy().mean(axis=0))

            result.write('\n Epoch: [{0}]: Validation....\n'.format(epoch_num + 1))
            result.write("\n========================================\n")
            result.write('euclidean_dist: {euclidean_dist:.4f}\t'
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
            result.write("\n------------------------------------------------------------------\n")
            result.write("Validation set results:\n")
            result.write('Multilabel metrics: example-based-classification:\n')
            result.write(
                         'hammingLoss: {hammingLoss:.4f}\t'
                         'accuracy: {accuracy:.4f}\t'
                         'precision: {precision:.4f}\t'
                         'recall: {recall:.4f}\t'
                         'fbeta: {fbeta:.4f}\t'.format(
                                                       hammingLoss=val_hammingLoss,
                                                       accuracy=val_eb_accuracy, precision=val_eb_precision,
                                                       recall=val_eb_recall, fbeta=val_eb_fbeta))


            result.write("\n")
            result.write('Multilabel metrics: example-based-ranking:\n')
            result.write('oneError: {oneError:.4f}\t'
                         'averagePrecision: {averagePrecision:.4f}\t'
                         'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=val_oneError,
                                                                   averagePrecision=val_averagePrecision,
                                                                   rankingLoss=val_rankingLoss))

            result.write("\n")
            result.write('Multilabel metrics: label-based-classification:\n')
            result.write('accuracyMacro: {accuracyMacro:.4f}\t'
                         'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=val_accuracyMacro,
                                                                 fbetaMicro=val_fbetaMicro))

            # result.write("\n")
            # result.write('Multilabel metrics: example-based-ranking:\n')
            # result.write('oneError: {oneError:.4f}\t'
            #              'coverage: {coverage:.4f}\t'
            #              'averagePrecision: {averagePrecision:.4f}\t'
            #              'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=val_oneError,
            #                                                        coverage=val_coverage,
            #                                                        averagePrecision=val_averagePrecision,
            #                                                        rankingLoss=val_rankingLoss))
            #
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
            #                                                      recallMacro=val_recallMacro,
            #                                                      recallMicro=val_recallMicro,
            #                                                      fbetaMacro=val_fbetaMacro,
            #                                                      fbetaMicro=val_fbetaMicro))
            #
            # result.write("\n")
            # result.write('Multilabel metrics: label-based-ranking:\n')
            # result.write('aucMacro: {aucMacro:.4f}\t'
            #              'aucMicro: {aucMicro:.4f}\t'
            #              'aucInstance: {aucInstance:.4f}\t'.format(aucMacro=val_aucMacro,
            #                                                        aucMicro=val_aucMicro,
            #                                                        aucInstance=val_aucInstance))

        # Pearson correlation
        emopcc += pearsonr(dis_prediction.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
    # 每一个epoch loss平均
    epoch_loss = val_loss / len(valSet)
    # 每一个epoch pcc平均
    epoch_pcc = emopcc / len(valSet)
    epoch_cosine = val_cosine / cnt
    epoch_accuracy = val_accuracy / cnt
    print('********',len(valSet))
    print('********',cnt)
    # validation loss
    val_loss = epoch_loss
    print("Validation: Epoch emotion distribution val_loss:", val_loss, "\nEpoch emotion distribution PCC:",
          epoch_pcc.item(), "\n", "==========================")
    result.write("\n------------------------------------------------------------------\n")
    result.write('Epoch: [{0}]\t' "Validation: Epoch emotion distribution val_loss: {val_loss: .4f}\t"
                 "\nEpoch emotion distribution PCC: {PCC: .4f}\t".format(epoch_num +1, val_loss=val_loss, PCC=epoch_pcc))
    result.write('\n*****************************Epoch: [{0}]\t end************************\n'.format(epoch_num +1))

    # checkpoint
    checkpoint = {
        'epoch': epoch_num + 1,
        'valid_loss_min_kl': epoch_loss,
        # 'valid_cosine_kl': epoch_cosine,
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
        print('Validation accuracy creased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_accuracy_min, epoch_accuracy))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path + "/train_co_attn_multi_dis_current_checkpoint.pt",
                 best_model_path + "/train_co_attn_multi_dis_best_model.pt")
        epoch_accuracy_min = epoch_accuracy

    print('\n*****************************Epoch: [{0}]\t end************************\n'.format(epoch_num +1))


# testing
net = MovieNet(args)
net, optimizer, start_epoch, valid_loss_min_kl = load_ckp(
    best_model_path + "/train_co_attn_multi_dis_best_model.pt", net, optimizer)
net.eval()
test_kl = 0
emopcc = 0

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

# example-based-ranking
sum_oneError = 0
sum_averagePrecision = 0
sum_rankingLoss = 0

# label-based-classification
sum_accuracyMacro = 0
sum_fbetaMicro = 0

# # example-based-ranking
# sum_oneError = 0
# sum_coverage = 0
# sum_averagePrecision = 0
# sum_rankingLoss = 0
#
# # label-based-classification
# sum_accuracyMacro = 0
# sum_accuracyMicro = 0
# sum_precisionMacro = 0
# sum_precisionMicro = 0
# sum_recallMacro = 0
# sum_recallMicro = 0
# sum_fbetaMacro = 0
# sum_fbetaMicro = 0
#
# # label-based-ranking
# sum_aucMacro = 0
# sum_aucMicro = 0
# sum_aucInstance = 0
count = 0
for i, data in enumerate(testDataloader):
    count = count + 1
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

    # Forward pass
    predict, input_clstm, shared_encoder, att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10 = \
        net(test, Frontal, Temporal, Central, Parietal, Occipital, dis)
    # print(att_1, att_2, att_3, att_4, att_5, att_6, att_7, att_8, att_9, att_10)

    #for loss1
    softmax = torch.nn.Softmax(dim=1)
    dis_prediction = softmax(predict)
    dis_prediction = dis_prediction.squeeze(dim=0)# [32,9]
    dis = torch.squeeze(dis, dim=1)
    loss1 = kl_div(dis_prediction.log(), dis)
    #for loss2
    label_prediction = torch.sigmoid(predict)
    #loss2: BCELoss
    dom_label = dom_label.detach()
    loss2 = mutilabel_criterion(label_prediction, dom_label)
    #loss2: MLSML
    # loss2 = MLSML(label_prediction.cuda(), target_gt.cuda())
    loss = lamda * loss1 + (1 - lamda) * loss2
    test_loss = loss.item()
    test_loss += test_loss / dis.shape[0]

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
    test_hammingLoss = hammingLoss(dom_label, label_prediction)
    sum_hammingLoss += test_hammingLoss
    test_eb_accuracy = accuracy(dom_label, label_prediction)
    sum_eb_accuracy += test_eb_accuracy
    test_eb_precision = precision(dom_label, label_prediction)
    sum_eb_precision += test_eb_precision
    test_eb_recall = recall(dom_label, label_prediction)
    sum_eb_recall += test_eb_recall
    test_eb_fbeta = fbeta(dom_label, label_prediction)
    sum_eb_fbeta += test_eb_fbeta

    # example-based-ranking
    test_oneError = oneError(dom_label, label_prediction)
    sum_oneError += test_oneError
    test_averagePrecision = averagePrecision(dom_label, label_prediction)
    sum_averagePrecision += test_averagePrecision
    test_rankingLoss = rankingLoss(dom_label, label_prediction)
    sum_rankingLoss += test_rankingLoss

    # label-based-classification
    test_accuracyMacro = accuracyMacro(dom_label, label_prediction)
    sum_accuracyMacro += test_accuracyMacro
    test_fbetaMicro = fbetaMicro(dom_label, label_prediction)
    sum_fbetaMicro += test_fbetaMicro

    # # example-based-ranking
    # test_oneError = oneError(dom_label, label_prediction)
    # sum_oneError += test_oneError
    # test_coverage = coverage(dom_label, label_prediction)
    # sum_coverage += test_coverage
    # test_averagePrecision = averagePrecision(dom_label, label_prediction)
    # sum_averagePrecision += test_averagePrecision
    # test_rankingLoss = rankingLoss(dom_label, label_prediction)
    # sum_rankingLoss += test_rankingLoss
    #
    # # label-based-classification
    # test_accuracyMacro = accuracyMacro(dom_label, label_prediction)
    # sum_accuracyMacro += test_accuracyMacro
    # test_accuracyMicro = accuracyMicro(dom_label, label_prediction)
    # sum_accuracyMicro += test_accuracyMicro
    # test_precisionMacro = precisionMacro(dom_label, label_prediction)
    # sum_precisionMacro += test_precisionMacro
    # test_precisionMicro = precisionMicro(dom_label, label_prediction)
    # sum_precisionMicro += test_precisionMicro
    # test_recallMacro = recallMacro(dom_label, label_prediction)
    # sum_recallMacro += test_recallMacro
    # test_recallMicro = recallMicro(dom_label, label_prediction)
    # sum_recallMicro += test_recallMicro
    # test_fbetaMacro = fbetaMacro(dom_label, label_prediction)
    # sum_fbetaMacro += test_recallMacro
    # test_fbetaMicro = fbetaMicro(dom_label, label_prediction)
    # sum_fbetaMicro += test_fbetaMicro
    #
    # # label-based-ranking
    # test_aucMacro = aucMacro(dom_label, label_prediction)
    # sum_aucMacro += test_aucMacro
    # test_aucMicro = aucMicro(dom_label, label_prediction)
    # sum_aucMicro += test_aucMicro
    # test_aucInstance = aucInstance(dom_label, label_prediction)
    # sum_aucInstance += test_aucInstance

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

        print("Test set results:\n",
              'Multilabel metrics: example-based-classification:\n',
              "hammingLoss= {:.4f}".format(test_hammingLoss),
              "accuracy= {:.4f}".format(test_eb_accuracy),
              "precision= {:.4f}".format(test_eb_precision),
              "recall= {:.4f}".format(test_eb_recall),
              "fbeta= {:.4f}".format(test_eb_fbeta))

        print("Test set results:\n",
              'Multilabel metrics: example-based-ranking:\n',
              "oneError= {:.4f}".format(test_oneError),
              "averagePrecision= {:.4f}".format(test_averagePrecision),
              "rankingLoss= {:.4f}".format(test_rankingLoss))

        print("Test set results:\n",
              'Multilabel metrics: label-based-classification:\n',
              "accuracyMacro= {:.4f}".format(test_accuracyMacro),
              "fbetaMicro= {:.4f}".format(test_fbetaMicro))

        # print("Test set results:\n",
        #       'Multilabel metrics: example-based-ranking:\n',
        #       "oneError= {:.4f}".format(test_oneError),
        #       "coverage= {:.4f}".format(test_coverage),
        #       "averagePrecision= {:.4f}".format(test_averagePrecision),
        #       "rankingLoss= {:.4f}".format(test_rankingLoss))
        #
        # print("Test set results:\n",
        #       'Multilabel metrics: label-based-classification:\n',
        #       "accuracyMacro= {:.4f}".format(test_accuracyMacro),
        #       "accuracyMicro= {:.4f}".format(test_accuracyMicro),
        #       "precisionMacro= {:.4f}".format(test_precisionMacro),
        #       "precisionMicro= {:.4f}".format(test_precisionMicro),
        #       "recallMacro= {:.4f}".format(test_recallMacro),
        #       "recallMicro= {:.4f}".format(test_recallMicro),
        #       "fbetaMacro= {:.4f}".format(test_fbetaMacro),
        #       "fbetaMicro= {:.4f}".format(test_fbetaMicro))
        #
        # print("Test set results:\n",
        #       'Multilabel metrics: label-based-ranking:\n',
        #       "aucMacro= {:.4f}".format(test_aucMacro),
        #       "aucMicro= {:.4f}".format(test_aucMicro),
        #       "aucInstance={:.4f}".format(test_aucInstance))

        result.write("\n------------------------------------------------------------------\n")
        result.write('Test co-attention:\n ')
        result.write('att_1: \t')
        result.write('%s\n' % att_1.cpu().detach().numpy().mean(axis=0))
        result.write('att_2: \t')
        result.write('%s\n' % att_2.cpu().detach().numpy().mean(axis=0))
        result.write('att_3: \t')
        result.write('%s\n' % att_3.cpu().detach().numpy().mean(axis=0))
        result.write('att_4: \t')
        result.write('%s\n' % att_4.cpu().detach().numpy().mean(axis=0))
        result.write('att_5: \t')
        result.write('%s\n' % att_5.cpu().detach().numpy().mean(axis=0))
        result.write('att_6: \t')
        result.write('%s\n' % att_6.cpu().detach().numpy().mean(axis=0))
        result.write('att_7: \t')
        result.write('%s\n' % att_7.cpu().detach().numpy().mean(axis=0))
        result.write('att_8: \t')
        result.write('%s\n' % att_8.cpu().detach().numpy().mean(axis=0))
        result.write('att_9: \t')
        result.write('%s\n' % att_9.cpu().detach().numpy().mean(axis=0))
        result.write('att_10: \t')
        result.write('%s\n' % att_10.cpu().detach().numpy().mean(axis=0))

        result.write('\n Epoch: [{0}]: Test....\n'.format(epoch_num+1))
        result.write("\n========================================\n")
        result.write('euclidean_dist: {euclidean_dist:.4f}\t'
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
        result.write("\n------------------------------------------------------------------\n")
        result.write("Test set results:\n")
        result.write('Multilabel metrics: example-based-classification:\n')
        result.write(
                     'hammingLoss: {hammingLoss:.4f}\t'
                     'accuracy: {accuracy:.4f}\t'
                     'precision: {precision:.4f}\t'
                     'recall: {recall:.4f}\t'
                     'fbeta: {fbeta:.4f}\t'.format(
                                                   hammingLoss=test_hammingLoss,
                                                   accuracy=test_eb_accuracy, precision=test_eb_precision,
                                                   recall=test_eb_recall, fbeta=test_eb_fbeta))
        result.write("\n")
        result.write('Multilabel metrics: example-based-ranking:\n')
        result.write('oneError: {oneError:.4f}\t'
                     'averagePrecision: {averagePrecision:.4f}\t'
                     'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=test_oneError,
                                                               averagePrecision=test_averagePrecision,
                                                               rankingLoss=test_rankingLoss))
        result.write("\n")
        result.write('Multilabel metrics: label-based-classification:\n')
        result.write('accuracyMacro: {accuracyMacro:.4f}\t'
                     'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=test_accuracyMacro,
                                                             fbetaMicro=test_fbetaMicro))

        # result.write("\n")
        # result.write('Multilabel metrics: example-based-ranking:\n')
        # result.write('oneError: {oneError:.4f}\t'
        #              'coverage: {coverage:.4f}\t'
        #              'averagePrecision: {averagePrecision:.4f}\t'
        #              'rankingLoss: {rankingLoss:.4f}\t'.format(oneError=test_oneError,
        #                                                        coverage=test_coverage,
        #                                                        averagePrecision=test_averagePrecision,
        #                                                        rankingLoss=test_rankingLoss))
        # result.write("\n")
        # result.write('Multilabel metrics: label-based-classification:\n')
        # result.write('accuracyMacro: {accuracyMacro:.4f}\t'
        #              'accuracyMicro: {accuracyMicro:.4f}\t'
        #              'precisionMacro: {precisionMacro:.4f}\t'
        #              'precisionMicro: {precisionMicro:.4f}\t'
        #              'recallMacro: {recallMacro:.4f}\t'
        #              'recallMicro: {recallMicro:.4f}\t'
        #              'fbetaMacro: {fbetaMacro:.4f}\t'
        #              'fbetaMicro: {fbetaMicro:.4f}\t'.format(accuracyMacro=test_accuracyMacro,
        #                                                      accuracyMicro=test_accuracyMicro,
        #                                                      precisionMacro=test_precisionMacro,
        #                                                      precisionMicro=test_precisionMicro,
        #                                                      recallMacro=test_recallMacro,
        #                                                      recallMicro=test_recallMicro,
        #                                                      fbetaMacro=test_fbetaMacro,
        #                                                      fbetaMicro=test_fbetaMicro))
        # result.write("\n")
        # result.write('Multilabel metrics: label-based-ranking:\n')
        # result.write('aucMacro: {aucMacro:.4f}\t'
        #              'aucMicro: {aucMicro:.4f}\t'
        #              'aucInstance: {aucInstance:.4f}\t'.format(aucMacro=test_aucMacro,
        #                                                        aucMicro=test_aucMicro,
        #                                                        aucInstance=test_aucInstance))

    # pearson correlation
    emopcc += pearsonr(dis_prediction.cpu().detach().numpy(), dis.cpu().detach().numpy())[0]
# average loss
test_testkl = test_loss / len(testSet)
# average pcc
test_emopcc = emopcc / len(testSet)
ave_euclidean = sum_euclidean/count
ave_chebyshev = sum_chebyshev/count
ave_kldist = sum_kldist/count
ave_clark = sum_clark/count
ave_canberra = sum_canberra/count
ave_cosine = sum_cosine/count
ave_intersection = sum_intersection/count

ave_hammingLoss = sum_hammingLoss/count
ave_eb_accuracy = sum_eb_accuracy/count
ave_eb_precision = sum_eb_precision/count
ave_eb_recall = sum_eb_recall/count
ave_eb_fbeta = sum_eb_fbeta/count

# example-based-ranking
ave_oneError = sum_oneError/count
ave_averagePrecision = sum_averagePrecision/count
ave_rankingLoss = sum_rankingLoss/count

# label-based-classification
ave_accuracyMacro = sum_accuracyMacro/count
ave_fbetaMicro = sum_fbetaMicro/count

# # example-based-ranking
# ave_oneError = sum_oneError/count
# ave_coverage = sum_coverage/count
# ave_averagePrecision = sum_averagePrecision/count
# ave_rankingLoss = sum_rankingLoss/count
#
# # label-based-classification
# ave_accuracyMacro = sum_accuracyMacro/count
# ave_accuracyMicro = sum_accuracyMicro/count
# ave_precisionMacro = sum_precisionMacro/count
# ave_precisionMicro = sum_precisionMicro/count
# ave_recallMacro = sum_recallMacro/count
# ave_recallMicro = sum_recallMicro/count
# ave_fbetaMacro = sum_fbetaMacro/count
# ave_fbetaMicro = sum_fbetaMicro/count
#
# # label-based-ranking
# ave_aucMacro = sum_aucMacro/count
# ave_aucMicro = sum_aucMicro/count
# ave_aucInstance = sum_aucInstance/count
print("\n================================================================================\n")
print("Test Emotion distribution test_loss:", test_testkl, "\Test Emotion distribution PCC:", test_emopcc.item())
result.write("\n------------------------------------------------------------------\n")
result.write('Epoch: [{0}]\t' "Test Emotion distribution test_loss:{test_loss: .4f}\n"
             "Test Emotion distribution PCC:{PCC: .4f}\t".format(epoch_num+1, test_loss=test_testkl,PCC=test_emopcc))

result.write("\n================================================================================\n")
result.write('Epoch: {epoch:.1f}\t'
                     'Test epoch_euclidean: {epoch_euclidean:.4f}\t'
                     'Test epoch_chebyshev: {epoch_chebyshev:.4f}\t'
                     'Test epoch_kldist: {epoch_kldist:.4f}\t'
                     'Test epoch_clark_dist: {epoch_clark_dist:.4f}\t'
                     'Test epoch_canberra_dist: {epoch_canberra_dist:.4f}\t'
                     'Test epoch_cosine_similarity: {epoch_cosine_similarity:.4f}\t'
                     'Test epoch_intersection_similarity: {epoch_intersection_similarity:.4f}\t'.format(epoch=epoch_num+1,
                                                                                                        epoch_euclidean=ave_euclidean,
                                                                                                        epoch_chebyshev=ave_chebyshev,
                                                                                                        epoch_kldist=ave_kldist,
                                                                                                        epoch_clark_dist=ave_clark,
                                                                                                        epoch_canberra_dist=ave_canberra,
                                                                                                        epoch_cosine_similarity=ave_cosine,
                                                                                                        epoch_intersection_similarity=ave_intersection))

#multi-label classification
result.write("\n================================================================================\n")
result.write('Epoch: {epoch:.1f}\t'
                     'Test epoch_accuracy: {epoch_accuracy:.4f}\t'
                     'Test epoch_eb_accuracy: {epoch_eb_accuracy:.4f}\t'
                     'Test epoch_eb_precision: {epoch_eb_precision:.4f}\t'
                     'Test epoch_eb_recall: {epoch_eb_recall:.4f}\t'
                     'Test epoch_eb_fbeta: {epoch_eb_fbeta:.4f}\t'
                     'Test epoch_hammingloss: {epoch_hammingloss:.4f}\t'
                     'Test epoch_oneError: {epoch_oneError:.4f}\t'
                     'Test epoch_Averageprecision: {epoch_Averageprecision:.4f}\t'
                     'Test epoch_rankingloss: {epoch_rankingloss:.4f}\t'
                     'Test epoch_accuracyMacro: {epoch_accuracyMacro:.4f}\t'
                     'Test epoch_fbetaMicro: {epoch_fbetaMicro:.4f}\t'.format(epoch=epoch_num+1,
                                                                            epoch_accuracy=epoch_accuracy,
                                                                            epoch_eb_accuracy=ave_eb_accuracy,
                                                                            epoch_eb_precision=ave_eb_precision,
                                                                            epoch_eb_recall=ave_eb_recall,
                                                                            epoch_eb_fbeta=ave_eb_fbeta,
                                                                            epoch_hammingloss=ave_hammingLoss,
                                                                            epoch_oneError=ave_oneError,
                                                                            epoch_Averageprecision=ave_averagePrecision,
                                                                            epoch_rankingloss=ave_rankingLoss,
                                                                            epoch_accuracyMacro=ave_accuracyMacro,
                                                                            epoch_fbetaMicro=ave_fbetaMicro))


print(att_1.cpu().detach().numpy().mean(axis=0), att_2.cpu().detach().numpy().mean(axis=0),
      att_3.cpu().detach().numpy().mean(axis=0), att_4.cpu().detach().numpy().mean(axis=0),
      att_5.cpu().detach().numpy().mean(axis=0), att_6.cpu().detach().numpy().mean(axis=0),
      att_7.cpu().detach().numpy().mean(axis=0), att_8.cpu().detach().numpy().mean(axis=0),
      att_9.cpu().detach().numpy().mean(axis=0), att_10.cpu().detach().numpy().mean(axis=0))

import csv

with open("GC_POSITIVE.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(GC_est)

