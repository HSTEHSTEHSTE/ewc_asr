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
import pickle

from utils import EWC, ewc_train, normal_train, test
from mlp import MLP

from nnet_models import nnetRNN
from datasets import nnetLoad
from copy import deepcopy
from tensorflow.contrib.framework.python.ops.variables import variable

# hyper parameters
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
    "lr_threshold": .00000001
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
train_size = 4000000
dev_size = 1000000
test_size = 40000

size_0 = 5855864
size_1 = 5874699
size_2 = 5862658
size_3 = 5880272
size_4 = 5856891

dataset_domain0 = nnetLoad(os.path.join(config.egs_dir, config.train_set), 'chunk_0.pt')
dataset_train_domain0, dataset_dev_domain0, dataset_test_domain0, remainder = torch.utils.data.random_split(dataset_domain0, [train_size, dev_size, test_size, (size_0 - train_size - dev_size - test_size)])
data_loader_train_domain0 = torch.utils.data.DataLoader(dataset_train_domain0, batch_size=config.batch_size, shuffle=True)
data_loader_test_domain0 = torch.utils.data.DataLoader(dataset_test_domain0, batch_size=config.batch_size, shuffle=True)
data_loader_dev_domain0 = torch.utils.data.DataLoader(dataset_dev_domain0, batch_size=config.batch_size, shuffle=True)
dataset_domain1 = nnetLoad(os.path.join(config.egs_dir, config.train_set), 'chunk_1.pt')
dataset_train_domain1, dataset_dev_domain1, dataset_test_domain1, remainder = torch.utils.data.random_split(dataset_domain1, [train_size, dev_size, test_size, (size_1 - train_size - dev_size - test_size)])
data_loader_train_domain1 = torch.utils.data.DataLoader(dataset_train_domain1, batch_size=config.batch_size, shuffle=True)
data_loader_test_domain1 = torch.utils.data.DataLoader(dataset_test_domain1, batch_size=config.batch_size, shuffle=True)
data_loader_dev_domain1 = torch.utils.data.DataLoader(dataset_dev_domain1, batch_size=config.batch_size, shuffle=True)
dataset_domain2 = nnetLoad(os.path.join(config.egs_dir, config.train_set), 'chunk_2.pt')
dataset_train_domain2, dataset_dev_domain2, dataset_test_domain2, remainder = torch.utils.data.random_split(dataset_domain2, [train_size, dev_size, test_size, (size_2 - train_size - dev_size - test_size)])
data_loader_train_domain2 = torch.utils.data.DataLoader(dataset_train_domain2, batch_size=config.batch_size, shuffle=True)
data_loader_test_domain2 = torch.utils.data.DataLoader(dataset_test_domain2, batch_size=config.batch_size, shuffle=True)
data_loader_dev_domain2 = torch.utils.data.DataLoader(dataset_dev_domain2, batch_size=config.batch_size, shuffle=True)
dataset_domain3 = nnetLoad(os.path.join(config.egs_dir, config.train_set), 'chunk_3.pt')
dataset_train_domain3, dataset_dev_domain3, dataset_test_domain3, remainder = torch.utils.data.random_split(dataset_domain3, [train_size, dev_size, test_size, (size_3 - train_size - dev_size - test_size)])
data_loader_train_domain3 = torch.utils.data.DataLoader(dataset_train_domain3, batch_size=config.batch_size, shuffle=True)
data_loader_test_domain3 = torch.utils.data.DataLoader(dataset_test_domain3, batch_size=config.batch_size, shuffle=True)
data_loader_dev_domain3 = torch.utils.data.DataLoader(dataset_dev_domain3, batch_size=config.batch_size, shuffle=True)
dataset_domain4 = nnetLoad(os.path.join(config.egs_dir, config.train_set), 'chunk_4.pt')
dataset_train_domain4, dataset_dev_domain4, dataset_test_domain4, remainder = torch.utils.data.random_split(dataset_domain4, [train_size, dev_size, test_size, (size_4 - train_size - dev_size - test_size)])
data_loader_train_domain4 = torch.utils.data.DataLoader(dataset_train_domain4, batch_size=config.batch_size, shuffle=True)
data_loader_test_domain4 = torch.utils.data.DataLoader(dataset_test_domain4, batch_size=config.batch_size, shuffle=True)
data_loader_dev_domain4 = torch.utils.data.DataLoader(dataset_dev_domain4, batch_size=config.batch_size, shuffle=True)

model_dir = os.path.join(config.store_path, config.experiment_name + '.dir')

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(model_dir, config.experiment_name),
    filemode='w')

# define a new Handler to log to console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

ep_start = 0
err_p = 10000000
model_path = os.path.join(model_dir, config.experiment_name + '__epoch_0.model')
torch.save({
    'epoch': 1000,
    'model_state_dict': model.state_dict(),
    'err_p': 10000000,
    'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))
debug_cutoff = 1024

# Set environment variable for GPU ID
id = get_device_id()
os.environ["CUDA_VISIBLE_DEVICES"] = id

model = model.cuda()

ep_loss_tr = []
ep_fer_tr = []
ep_loss_dev = []
ep_fer_dev = []
best_model_state = model.state_dict()

criterion = nn.CrossEntropyLoss()

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def reset_runtime():
    lr = .001
    lr_below_threshold = False
    epoch_i = ep_start

def train(train_domain, test_domain_current, test_domain_previous_list):
    reset_runtime()
    while epoch_i < config.epochs and not lr_below_threshold:
        print("epoch: ", epoch_i)
        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        tr_fer = []

        batch_l = torch.tensor([117] * config.batch_size)

        # Main training loop
        for n, (batch_x, lab) in enumerate(data_loader_train_domain0):

            if n > debug_cutoff:
                break

            # todo: handle last batch
            indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
            lab = torch.unsqueeze(lab, 1).long()
            if config.use_gpu:
                batch_x = Variable(batch_x).cuda()
                lab = Variable(lab).cuda()
            else:
                batch_x = Variable(batch_x)
                lab = Variable(lab)

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)
            
            loss = criterion(class_out, lab)

            train_losses.append(loss.item())
            if config.use_gpu:
                tr_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                tr_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
            optimizer.step()

        ep_loss_tr.append(np.mean(train_losses))
        ep_fer_tr.append(np.mean(tr_fer))

        ######################
        ##### Validation #####
        ######################
        model.eval()
        val_losses = []
        val_fer = []
        
        # Main training loop
        for (n, (batch_x, lab)) in enumerate(data_loader_train_domain0):
            if n > debug_cutoff:
                break

            # _, indices = torch.sort(batch_l, descending=True)
            # if config.use_gpu:
            #     batch_x = Variable(batch_x[indices]).cuda()
            #     batch_l = Variable(batch_l[indices]).cuda()
            #     lab = Variable(lab[indices]).cuda()
            # else:
            #     batch_x = Variable(batch_x[indices])
            #     batch_l = Variable(batch_l[indices])
            #     lab = Variable(lab[indices])
            indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
            lab = torch.unsqueeze(lab, 1).long()
            if config.use_gpu:
                batch_x = Variable(batch_x).cuda()
                lab = Variable(lab).cuda()
            else:
                batch_x = Variable(batch_x)
                lab = Variable(lab)

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)

            loss = criterion(class_out, lab)

            val_losses.append(loss.item())
            if config.use_gpu:
                val_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
            else:
                val_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

        # Manage learning rate and revert model
        if epoch_i == 0:
            err_p = np.mean(val_losses)
            best_model_state = model.state_dict()
        else:
            if np.mean(val_losses) > (100 - config.lr_tol) * err_p / 100:
                logging.info(
                    "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
                lr = config.lrr * lr
                if lr < config.lr_threshold:
                    lr_below_threshold = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                model.load_state_dict(best_model_state)
            else:
                err_p = np.mean(val_losses)
                best_model_state = model.state_dict()

        ep_loss_dev.append(np.mean(val_losses))
        ep_fer_dev.append(np.mean(val_fer))

        print_log = "Epoch: {:d} ((lr={:.6f})) Tr loss: {:.3f} :: Tr FER: {:.2f}".format(epoch_i + 1, lr,
                                                                                            ep_loss_tr[-1],
                                                                                            ep_fer_tr[-1])
        print_log += " || Val: {:.3f} :: Val FER: {:.2f}".format(ep_loss_dev[-1], ep_fer_dev[-1])
        logging.info(print_log)

        if (epoch_i + 1) % config.model_save_interval == 0:
            model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
            torch.save({
                'epoch': epoch_i + 1,
                'feature_dim': config.feature_dim,
                'num_frames': num_frames,
                'num_classes': config.num_classes,
                'num_layers': config.num_layers,
                'hidden_dim': config.hidden_dim,
                'ep_loss_tr': ep_loss_tr,
                'ep_loss_dev': ep_loss_dev,
                'dropout': config.dropout,
                'lr': lr,
                'err_p': err_p,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))


        total_loss = 0
        total_accurate = torch.tensor(0.0).cuda()
        for (n, (batch_x, lab)) in enumerate(data_loader_test_domain0):
            if n > debug_cutoff:
                break
            indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
            lab = torch.unsqueeze(lab, 1).long()
            if config.use_gpu:
                batch_x = Variable(batch_x).cuda()
                lab = Variable(lab).cuda()
            else:
                batch_x = Variable(batch_x)
                lab = Variable(lab)

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l).cuda()
            loss = criterion(class_out, lab)
            class_out_lab = torch.argmax(class_out, dim = 1).cuda()
            accuracy_tensor = torch.where(class_out_lab - lab < .5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
            additional_accuracy = torch.sum(accuracy_tensor)
            total_accurate += additional_accuracy
            total_loss += loss.item()
        
        accuracies_0_0.append(total_accurate / test_size)
        losses_0_0.append(total_loss / test_size)

        epoch_i += 1

# *********************************** Domain 0 *********************************** #
accuracies_0_0 = []
losses_0_0 = []

train(data_loader_train_domain0, data_loader_dev_domain0, [data_loader_test_domain0])

plt.plot(accuracies_0_0)
plt.savefig('accuracies_0_0.png')
plt.clf()

plt.plot(losses_0_0)
plt.savefig('losses_0_0.png')
plt.clf()

# *********************************** Domain 1 *********************************** #

accuracies_0_1 = []
losses_0_1 = []
accuracies_1_1 = []
losses_1_1 = []

reset_runtime()

while epoch_i < config.epochs and not lr_below_threshold:

    print("epoch: ", epoch_i)
    ####################
    ##### Training #####
    ####################

    model.train()
    train_losses = []
    tr_fer = []

    batch_l = torch.tensor([117] * config.batch_size)

    # Main training loop
    for n, (batch_x, lab) in enumerate(data_loader_train_domain1):

        if n > debug_cutoff:
            break

        # todo: handle last batch
        indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
        lab = torch.unsqueeze(lab, 1).long()
        if config.use_gpu:
            batch_x = Variable(batch_x).cuda()
            lab = Variable(lab).cuda()
        else:
            batch_x = Variable(batch_x)
            lab = Variable(lab)

        optimizer.zero_grad()
        # Main forward pass
        class_out = model(batch_x, batch_l)
        class_out = pad2list(class_out, batch_l)
        lab = pad2list(lab, batch_l)
        
        loss = criterion(class_out, lab)

        train_losses.append(loss.item())
        if config.use_gpu:
            tr_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
        else:
            tr_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_thresh)
        optimizer.step()

    ep_loss_tr.append(np.mean(train_losses))
    ep_fer_tr.append(np.mean(tr_fer))

    ######################
    ##### Validation #####
    ######################
    model.eval()
    val_losses = []
    val_fer = []
    
    # Main training loop
    for (n, (batch_x, lab)) in enumerate(data_loader_dev_domain1):
        if n > debug_cutoff:
            break

        # _, indices = torch.sort(batch_l, descending=True)
        # if config.use_gpu:
        #     batch_x = Variable(batch_x[indices]).cuda()
        #     batch_l = Variable(batch_l[indices]).cuda()
        #     lab = Variable(lab[indices]).cuda()
        # else:
        #     batch_x = Variable(batch_x[indices])
        #     batch_l = Variable(batch_l[indices])
        #     lab = Variable(lab[indices])
        indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
        lab = torch.unsqueeze(lab, 1).long()
        if config.use_gpu:
            batch_x = Variable(batch_x).cuda()
            lab = Variable(lab).cuda()
        else:
            batch_x = Variable(batch_x)
            lab = Variable(lab)

        optimizer.zero_grad()
        # Main forward pass
        class_out = model(batch_x, batch_l)
        class_out = pad2list(class_out, batch_l)
        lab = pad2list(lab, batch_l)

        loss = criterion(class_out, lab)

        val_losses.append(loss.item())
        if config.use_gpu:
            val_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
        else:
            val_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))

    # Manage learning rate and revert model
    if epoch_i == 0:
        err_p = np.mean(val_losses)
        best_model_state = model.state_dict()
    else:
        if np.mean(val_losses) > (100 - config.lr_tol) * err_p / 100:
            logging.info(
                "Val loss went up, Changing learning rate from {:.6f} to {:.6f}".format(lr, config.lrr * lr))
            lr = config.lrr * lr
            if lr < config.lr_threshold:
                lr_below_threshold = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            model.load_state_dict(best_model_state)
        else:
            err_p = np.mean(val_losses)
            best_model_state = model.state_dict()

    ep_loss_dev.append(np.mean(val_losses))
    ep_fer_dev.append(np.mean(val_fer))

    print_log = "Epoch: {:d} ((lr={:.6f})) Tr loss: {:.3f} :: Tr FER: {:.2f}".format(epoch_i + 1, lr,
                                                                                        ep_loss_tr[-1],
                                                                                        ep_fer_tr[-1])
    print_log += " || Val: {:.3f} :: Val FER: {:.2f}".format(ep_loss_dev[-1], ep_fer_dev[-1])
    logging.info(print_log)

    if (epoch_i + 1) % config.model_save_interval == 0:
        model_path = os.path.join(model_dir, config.experiment_name + '__epoch_%d' % (epoch_i + 1) + '.model')
        torch.save({
            'epoch': epoch_i + 1,
            'feature_dim': config.feature_dim,
            'num_frames': num_frames,
            'num_classes': config.num_classes,
            'num_layers': config.num_layers,
            'hidden_dim': config.hidden_dim,
            'ep_loss_tr': ep_loss_tr,
            'ep_loss_dev': ep_loss_dev,
            'dropout': config.dropout,
            'lr': lr,
            'err_p': err_p,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, (open(model_path, 'wb')))

    total_loss = 0
    total_accurate = torch.tensor(0.0).cuda()
    for (n, (batch_x, lab)) in enumerate(data_loader_test_domain1):
        if n > debug_cutoff:
            break
        indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
        lab = torch.unsqueeze(lab, 1).long()
        if config.use_gpu:
            batch_x = Variable(batch_x).cuda()
            lab = Variable(lab).cuda()
        else:
            batch_x = Variable(batch_x)
            lab = Variable(lab)

        optimizer.zero_grad()
        # Main forward pass
        class_out = model(batch_x, batch_l)
        class_out = pad2list(class_out, batch_l)
        lab = pad2list(lab, batch_l)
        lab = torch.argmax(lab)
        class_out_lab = torch.argmax(class_out, dim = 1).cuda()
        accuracy_tensor = torch.where(class_out_lab - lab < .5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        additional_accuracy = torch.sum(accuracy_tensor)
        total_accurate += additional_accuracy
        total_loss += loss.item()
    
    accuracies_1_1.append(total_accurate / test_size)
    losses_1_1.append(total_loss / test_size)

    total_loss = 0
    total_accurate = torch.tensor(0.0).cuda()
    for (n, (batch_x, lab)) in enumerate(data_loader_test_domain0):
        if n > debug_cutoff:
            break
        indices = torch.tensor(batch_l, dtype = torch.int64).cuda()
        lab = torch.unsqueeze(lab, 1).long()
        if config.use_gpu:
            batch_x = Variable(batch_x).cuda()
            lab = Variable(lab).cuda()
        else:
            batch_x = Variable(batch_x)
            lab = Variable(lab)

        optimizer.zero_grad()
        # Main forward pass
        class_out = model(batch_x, batch_l)
        class_out = pad2list(class_out, batch_l)
        lab = pad2list(lab, batch_l)
        lab = torch.argmax(lab)
        class_out_lab = torch.argmax(class_out, dim = 1).cuda()
        accuracy_tensor = torch.where(class_out_lab - lab < .5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        additional_accuracy = torch.sum(accuracy_tensor)
        total_accurate += additional_accuracy
        total_loss += loss.item()
    
    accuracies_0_1.append(total_accurate / (test_size))
    losses_0_1.append(total_loss / (test_size))

    epoch_i += 1

print(len(accuracies_0_0), len(accuracies_0_1))
xs_1 = np.arange(len(accuracies_0_0), len(accuracies_0_0) + len(accuracies_0_1))
accuracies_0_0 += accuracies_0_1

plt.plot(accuracies_0_0)
print(xs_1)
print(accuracies_0_0, accuracies_1_1)
plt.plot(xs_1, accuracies_1_1)
plt.savefig('accuracies_1_1.png')
plt.clf()
