import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import subprocess
from torch.autograd import Variable
import numpy as np
import logging

from utils import EWC, ewc_train, normal_train, test
from mlp import MLP

from nnet_models import nnetRNN
from datasets import nnetDatasetSeq
from copy import deepcopy
from tensorflow.contrib.framework.python.ops.variables import variable

# hyper parameters
epochs = 30
lr = .001
sample_size = 200
hidden_size = 200
num_task = 5
num_frames = 9
# ************************************* #

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

config = {
    "egs_dir": "/home/xli257/ASR",
    "store_path": "/home/xli257/ASR/store",
    "train_set": "train_si284",
    "num_layers": 5,
    "hidden_dim": 117,
    "feature_dim": 13,
    "epochs": 100,
    "learning_rate": .001,
    "dropout": 0,
    "num_classes": 42,
    "weight_decay": 0,
    "batch_size": 64,
    "experiment_name": "exp_run",
    "use_gpu": True,
    "clip_thresh": 1,
    "model_save_interval": 50,
    "lr_tol": .5,
    "lrr": .5,
}

config = AttributeDict(config)

def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]

def pad2list(padded_seq, lengths):
    return torch.cat([padded_seq[i, 0:lengths[i]] for i in range(padded_seq.size(0))])

def softmax(X):
    return np.exp(X) / np.tile(np.sum(np.exp(X), axis=1)[:, None], (1, X.shape[1]))

def compute_fer(x, l):
    x = softmax(x)
    preds = np.argmax(x, axis=1)
    err = (float(preds.shape[0]) - float(np.sum(np.equal(preds, l)))) * 100 / float(preds.shape[0])
    return err

model = nnetRNN(config.feature_dim * num_frames, config.num_layers, config.hidden_dim,
                    config.num_classes, config.dropout)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Load datasets
