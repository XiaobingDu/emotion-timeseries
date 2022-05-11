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
import math
import scipy.stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import hamming_loss

#___________________________________________________________________________________________________________________________

# New Dataloader for MovieClass
class MediaEvalDataset(Dataset):
    def __init__(self,feature, dis, dom_label, idx):
        self.channel = 30
        self.feature = feature
        self.dis = dis
        self.dom_label = dom_label
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
        target_gt = self.dom_label[index]

        # 将5个脑区的数据hstack
        # combined = np.hstack([F, T, C, P, O])
        combined = torch.cat((F, T, C, P, O), dim=-1) #[10,150]

        return combined, y, target_gt, F, T, C, P, O



#______________________________________________________________________________________________________________________________________
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr_decay = 10 equals to lr = lr * 0.1
    """
    if epoch == 5: #100
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

    return model, optimizer, checkpoint['epoch'], valid_loss_min_kl

## metrics for emotion distribution learning

# euclidean distance
def euclidean_dist (size, RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()
    _size = size
    dist = np.empty(_size)
    for i in range(_size):
        d1 = np.sqrt(np.sum(np.square(RD[i] - PD[i])))
        dist[i] = d1
    euclidean_dist = sum(dist)/_size

    return euclidean_dist

# chebyshev distance
def chebyshev_dist (size, RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()
    _size = size
    chebyshev_distances = np.empty(_size)
    for i in range(_size):
        chebyshev_distances[i] = np.max(np.abs(RD[i]-PD[i]))

    return sum(chebyshev_distances) / _size

# Kullback-Leibler divergence
def KL_dist(RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()

    kldiv = RD * np.log(RD / PD)
    kldist = kldiv.sum(axis=1)
    kldist = [x for x in kldist if str(x) != 'nan' and str(x) != 'inf']  # 除去inf值
    kldist = np.mean(kldist)

    return kldist

# clark distance
def clark_dist(RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()

    temp = PD - RD
    temp = temp * temp
    temp2 = PD + RD
    temp2 = temp2 * temp2
    temp = temp / temp2
    temp = temp.sum(axis=1)
    temp = np.sqrt(temp)
    distance = np.mean(temp)

    return distance

#canberra metric
def canberra_dist(RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()

    temp = np.abs(PD - RD)
    temp2 = PD + RD
    temp = temp / temp2
    temp = temp.sum(axis=1)
    distance = np.mean(temp)

    return distance

#cosine coefficient
def cosine_dist(RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()

    temp = PD * RD
    inner = temp.sum(axis=1)
    pd_temp = PD * PD
    pd_temp = pd_temp.sum(axis=1)
    rd_temp = RD * RD
    rd_temp = rd_temp.sum(axis=1)
    len = np.sqrt(pd_temp) * np.sqrt(rd_temp)
    len = [x for x in len if str(x) != 'nan' and str(x) != 'inf']  # 除去inf值
    tmp = inner/len
    tmp = [x for x in tmp if str(x) != 'nan' and str(x) != 'inf']
    distance = np.mean(tmp)

    return distance

#intersection similarity
def intersection_dist(RD, PD):
    RD = RD.cpu().detach().numpy()
    PD = PD.cpu().detach().numpy()

    temp = np.minimum(PD,RD)
    temp = temp.sum(axis=1)
    similarity = np.mean(temp)

    return similarity



##metrics for multi-label emotion predictation
## from CVPR 2019

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class(CP).
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output) # torch.from_numpy: transfer the 'numpy' to 'tensor'; same as torch.Tensor()
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        # the dim of output & target
        if output.dim() == 1:
            output = output.view(-1, 1) #add one dim
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0: # numel: the number of tensor
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class（CP）
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples) # get cp
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True): # for one class

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True) #sorted by column; indices: the index for the elements rank

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

#generate adj
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file,'rb'), encoding='iso-8859-1')
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