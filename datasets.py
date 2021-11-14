from os import listdir
from os.path import join
import pickle
import random
import numpy as np

import torch
from torch.utils import data


class nnetDataset(data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index, :], self.label[index]


class nnetLoad(data.Dataset):

    def __init__(self, path, name):
        self.path = path
        self.ids = [f for f in listdir(self.path) if f.endswith(name)]
        self.data = torch.load(join(self.path, name))
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1]

    def random_sample(self, number_k):
        all_indices = list(np.arange(0, self.data.shape[0]))
        indexing_list = random.sample(all_indices, k=number_k)
        return self.data[indexing_list, :-1], self.data[indexing_list, -1]

class nnetDatasetSeq(data.Dataset):

    def __init__(self, path, seq_len = 512, ids = None):
        self.path = path
        with open(join(path, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)
        self.labels = torch.load(join(self.path, 'labels.pkl'))
        if ids is None:
            self.ids = [f for f in listdir(self.path) if f.endswith('.pt')]  # list(self.labels.keys())
        else:
            self.ids = ids
        self.ids = [i for i in self.ids if i in list(self.labels.keys())]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = torch.load(join(self.path, self.ids[index]))
        l = self.lengths[self.ids[index]]
        l = max(l, self.seq_len)
        x = torch.nn.functional.pad(x, (0, 0, 0, self.seq_len - x.shape[0]), value = 0)
        x = x[:self.seq_len, :]
        lab = self.labels[self.ids[index]]
        lab = torch.nn.functional.pad(lab, (0, self.seq_len - lab.shape[0]), value = 0)
        lab = lab[:self.seq_len]
        return x, l, lab

    def random_sample(self, number_k):
        ids = random.sample(self.ids[:self.seq_len], k=number_k)
        return nnetDatasetSeq(self.path, self.seq_len, ids)

class nnetDataset3Seq(data.Dataset):

    def __init__(self, paths):
        self.path1 = paths[0]
        self.path2 = paths[1]
        self.path3 = paths[2]
        with open(join(self.path1, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)
        self.labels = torch.load(join(self.path1, 'labels.pkl'))
        self.ids = list(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x1 = torch.load(join(self.path1, self.ids[index]))
        x2 = torch.load(join(self.path2, self.ids[index]))
        x3 = torch.load(join(self.path3, self.ids[index]))
        l = self.lengths[self.ids[index]]
        lab = self.labels[self.ids[index]]
        return x1, x2, x3, l, lab


class nnetDatasetSeqAE(data.Dataset):

    def __init__(self, path):
        self.path = path
        self.ids = [f for f in listdir(self.path) if f.endswith('.pt')]
        with open(join(path, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = torch.load(join(self.path, self.ids[index]))
        l = self.lengths[self.ids[index]]
        return x, l