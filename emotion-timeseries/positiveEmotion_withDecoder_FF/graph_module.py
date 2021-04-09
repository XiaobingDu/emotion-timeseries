#-*-coding=utf-8-*-

from torch.nn import Parameter
import torch
import torch.nn as nn
import math
from utils_co_attn_GC import gen_A, gen_adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #显示属性
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCN, self).__init__()

        self.num_classes = num_classes

        #定义 GCN 2-layers
        self.hidden_size = 1024
        self.output_size = 2048
        self.gc1 = GraphConvolution(in_channel, self.hidden_size)
        self.gc2 = GraphConvolution(self.hidden_size, self.output_size)
        # self.gc1 = GraphConvolution(in_channel, 1024)
        # self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, inp):

        #GCN：learning inter-dependent object classification
        import pickle
        embedding = pickle.load(open(inp,'rb'), encoding='iso-8859-1')
        embedding = torch.Tensor(embedding)
        inp = embedding
        # tensor.detach(): 从self.A中分离出来的adj, 此时的adj与A共享存储空间
        #adj与self.A的区别：adj没有梯度，self.A有梯度；在adj没有改变的情况下self.A可以反向求导，adj不可以
        #adj是首先计算好的，在training过程中不会改变
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj) # gcn output

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]