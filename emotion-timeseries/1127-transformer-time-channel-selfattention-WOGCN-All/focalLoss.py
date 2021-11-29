# -*-coding:utf-8-*-

from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        print('*****', labels.dim())
        print (labels)
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()



#
# class FocalLoss(nn.Module):
#     def __init__(self, num_classes=9,  alpha=0.25, gamma=2, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param num_classes:     类别数量
#         :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param size_average:    损失计算方式,默认取均值
#         """
#         super(FocalLoss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
#         :return:
#         """
#
#         # 固定类别维度，其余合并(总检测框数或总批次数)，preds.size(-1)是最后一个维度
#         preds = preds.view(-1, preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#
#         # 使用log_softmax解决溢出问题，方便交叉熵计算而不用考虑值域
#         preds_logsoft = F.log_softmax(preds, dim=1)
#
#         # log_softmax是softmax+log运算，那再exp就算回去了变成softmax
#         preds_softmax = torch.exp(preds_logsoft)
#
#         # 这部分实现nll_loss ( crossentropy = log_softmax + nll)
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1).long())
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#
#         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#
#         # torch.mul 矩阵对应位置相乘，大小一致
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
#
#         # torch.t()求转置
#         loss = torch.mul(self.alpha, loss.t())
#         # print(loss.size()) [1,5]
#
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#
#         return loss