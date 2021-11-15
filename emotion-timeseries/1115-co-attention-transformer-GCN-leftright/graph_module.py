#-*-coding=utf-8-*-

from torch.nn import Parameter
import torch
import torch.nn as nn
import math
from utils import gen_A, gen_adj

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
    def __init__(self, num_classes, in_channel=300, t=0, adj_file=None): #t=0.4
        super(GCN, self).__init__()
        self.num_classes = num_classes
        #定义 GCN 2-layers
        self.hidden_size = 128 #1024
        self.output_size = 256 #2048
        #GCN 1-layer
        self.gc1 = GraphConvolution(in_channel, self.output_size)
        #GCN 2-layer
        # self.gc1 = GraphConvolution(in_channel, self.hidden_size)
        # self.gc2 = GraphConvolution(self.hidden_size, self.output_size)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, inp):

        #GCN：learning inter-dependent object classification
        import pickle
        embedding = pickle.load(open(inp,'rb'), encoding='iso-8859-1')
        embedding = torch.Tensor(embedding)
        inp = embedding
        adj = gen_adj(self.A).detach()
        #GCN 1-layer
        x = self.gc1(inp, adj)
        x = self.relu(x)
        #GCN 2-layer
        # x = self.gc1(inp, adj)
        # x = self.relu(x)
        # x = self.gc2(x, adj) # gcn output

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                # {'params': self.gc2.parameters(), 'lr': lr},
                ]