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

def regions(self,feature, dis, idx):

    return True