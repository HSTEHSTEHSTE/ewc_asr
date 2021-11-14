from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import tqdm
import time


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, config, seq_len = 512, device_id = None):

        self.model = model
        self.data_domain = dataset
        self.config = config
        self.nllloss = torch.nn.NLLLoss()
        self.seq_len = seq_len
        self.device_id = device_id

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}


        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p

        self.model.train()
        for x, l, lab in self.data_domain:
            self.model.zero_grad()
            input = variable(x)
            l = torch.clamp(l, max=self.seq_len)
            if self.device_id is not None:
                l.cuda()
                input.cuda()
            output = self.model(input, l).squeeze(0)
            label = torch.argmax(output, dim=1).cuda()
            loss = self.nllloss(F.log_softmax(output, dim=1), label)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / self.config.batch_size

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.train()
        return precision_matrices

    def update_model_weights(self, model: nn.Module):
        self.model = model
        self._precision_matrices = self._diag_fisher()

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct / float(len(data_loader.dataset))
