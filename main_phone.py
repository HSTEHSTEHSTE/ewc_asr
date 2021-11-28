import os
import matplotlib
from torch.utils import data
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
import time

from utils import CL, pad2list, softmax, compute_fer
from mlp import MLP

from nnet_models import nnetRNN
from datasets import nnetDatasetSeq
from copy import deepcopy
# from tensorflow.contrib.framework.python.ops.variables import variable

# hyper parameters
sample_size = 200
hidden_size = 200
num_frames = 1
# ************************************* #

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

config = {
    "egs_dir": "/export/c01/ssadhu/speech_confidence/e2e/wsj/data_for_henry/",
    "store_path": "/home/xli257/ASR/store",
    "train_set": "train_si284",
    "num_layers": 5,
    "hidden_dim": 512,
    "feature_dim": 13,
    "epochs": 100,
    "learning_rate": .001,
    "dropout": .3,
    "num_classes": 42,
    "weight_decay": 0,
    "batch_size": 64,
    "experiment_name": "exp_run",
    "use_gpu": True,
    "clip_thresh": 1,
    "model_save_interval": 10,
    "lr_tol": .5,
    "lrr": .5,
    "lr_threshold": .000001,
    "previous_random_sample_size": 5,
    "load_data_workers": 5,
    "load_checkpoint": None,
    "seq_len": 512,
    "cl_type": 'ewc'
}

config = AttributeDict(config)

def get_device_id():
    cmd = 'free-gpu'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return proc.stdout.decode('utf-8').strip().split()[0]

model = nnetRNN(config.feature_dim * num_frames, config.num_layers, config.hidden_dim,
                    config.num_classes, config.dropout)
# optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Load datasets
# 0: wsj, 1: reverb

dataset_train_domain0 = nnetDatasetSeq(os.path.join(config.egs_dir, 'wsj/train_si284'))
dataset_dev_domain0 = nnetDatasetSeq(os.path.join(config.egs_dir, 'wsj/test_dev93'))
dataset_test_domain0 = nnetDatasetSeq(os.path.join(config.egs_dir, 'wsj/test_eval92'))
data_loader_train_domain0 = torch.utils.data.DataLoader(dataset_train_domain0, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)
data_loader_dev_domain0 = torch.utils.data.DataLoader(dataset_dev_domain0, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)
data_loader_test_domain0 = torch.utils.data.DataLoader(dataset_test_domain0, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)

dataset_train_domain1 = nnetDatasetSeq(os.path.join(config.egs_dir, 'reverb/tr_simu_8ch'))
dataset_dev_test_domain1 = nnetDatasetSeq(os.path.join(config.egs_dir, 'reverb/dt_simu_8ch')) # len = 11872
dataset_dev_domain1, dataset_test_domain1, _ = torch.utils.data.random_split(dataset_dev_test_domain1, [1500, 10000, 11872 - 11500])
data_loader_train_domain1 = torch.utils.data.DataLoader(dataset_train_domain1, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)
data_loader_dev_domain1 = torch.utils.data.DataLoader(dataset_dev_domain1, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)
data_loader_test_domain1 = torch.utils.data.DataLoader(dataset_test_domain1, batch_size=config.batch_size, shuffle=True, num_workers=config.load_data_workers)

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
debug_cutoff = 999999999

# Set environment variable for GPU ID
id = get_device_id()
os.environ["CUDA_VISIBLE_DEVICES"] = id

model = model.cuda()

ep_loss_tr = []
ep_fer_tr = []
ep_loss_dev = []
ep_fer_dev = []
best_model_state = model.state_dict()
fisher_matrices = []
fisher_matrix_norms = []

criterion = nn.CrossEntropyLoss()

outfile_0 = open("result_0", mode="wb")
outfile_1 = open("result_1", mode="wb")
matrix_file = open("matrix", mode="wb")
norm_file = open("norm", mode="wb")

def reset_runtime():
    lr = .001
    lr_below_threshold = False
    epoch_i = ep_start
    return lr, lr_below_threshold, epoch_i

def train(train_domain, dev_domain, test_domain_previous_list, previous_dataset_list = []):
    lr, lr_below_threshold, epoch_i = reset_runtime()
    accuracies = [[] for i in range(len(test_domain_previous_list))]
    losses = [[] for i in range(len(test_domain_previous_list))]
    
    old_tasks = []
    if len(previous_dataset_list) > 0:
        for previous_dataset in previous_dataset_list:
            old_tasks.append(previous_dataset.random_sample(config.previous_random_sample_size))
        old_task = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset(old_tasks), num_workers=config.load_data_workers)
        gpu_id = None
        if config.use_gpu:
            gpu_id = id
        cl_object = CL(model = model, dataset = old_task, config = config, criterion = criterion, seq_len = config.seq_len, device_id = gpu_id, type = config.cl_type)

    while epoch_i < config.epochs and not lr_below_threshold:
        print("epoch: ", epoch_i)
        ####################
        ##### Training #####
        ####################

        model.train()
        train_losses = []
        tr_fer = []

        # Main training loop
        for n, (batch_x, batch_l, lab) in enumerate(train_domain):

            if n > debug_cutoff:
                break

            batch_l = torch.clamp(batch_l, max=config.seq_len)

            # todo: handle last batch
            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()
                lab = Variable(lab[indices]).cuda()
            else:
                batch_x = Variable(batch_x[indices])
                batch_l = Variable(batch_l[indices])
                lab = Variable(lab[indices])

            optimizer.zero_grad()
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)
            loss = criterion(class_out, lab)
            
            if len(old_tasks) > 0:
                cl_object.update_model_weights(model)
                loss += 10 * cl_object.penalty(model) / lr

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

        if len(old_tasks) > 0:
            fisher_matrices.append(cl_object._precision_matrices)

        ######################
        ##### Validation #####
        ######################
        model.eval()
        val_losses = []
        val_fer = []
        
        # Main training loop
        for (n, (batch_x, batch_l, lab)) in enumerate(dev_domain):
            if n > debug_cutoff:
                break

            batch_l = torch.clamp(batch_l, max=config.seq_len)

            # _, indices = torch.sort(batch_l, descending=True)
            # if config.use_gpu:
            #     batch_x = Variable(batch_x[indices]).cuda()
            #     batch_l = Variable(batch_l[indices]).cuda()
            #     lab = Variable(lab[indices]).cuda()
            # else:
            #     batch_x = Variable(batch_x[indices])
            #     batch_l = Variable(batch_l[indices])
            #     lab = Variable(lab[indices])
            _, indices = torch.sort(batch_l, descending=True)
            if config.use_gpu:
                batch_x = Variable(batch_x[indices]).cuda()
                batch_l = Variable(batch_l[indices]).cuda()
                lab = Variable(lab[indices]).cuda()
            else:
                batch_x = Variable(batch_x[indices])
                batch_l = Variable(batch_l[indices])
                lab = Variable(lab[indices])

            optimizer.zero_grad()
            
            # Main forward pass
            class_out = model(batch_x, batch_l)
            class_out = pad2list(class_out, batch_l)
            lab = pad2list(lab, batch_l)

            loss = criterion(class_out, lab)

            if len(old_tasks) > 0:
                loss += 10 * cl_object.penalty(model) / lr

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

        for test_domain_number, test_domain in enumerate(test_domain_previous_list):
            total_loss = 0
            total_accurate = 0
            total_number = 0
            test_fer = []
            for (n, (batch_x, batch_l, lab)) in enumerate(test_domain):
                if n > debug_cutoff:
                    break

                batch_l = torch.clamp(batch_l, max=config.seq_len)

                _, indices = torch.sort(batch_l, descending=True)
                if config.use_gpu:
                    batch_x = Variable(batch_x[indices]).cuda()
                    batch_l = Variable(batch_l[indices]).cuda()
                    lab = Variable(lab[indices]).cuda()
                else:
                    batch_x = Variable(batch_x[indices])
                    batch_l = Variable(batch_l[indices])
                    lab = Variable(lab[indices])

                optimizer.zero_grad()
                # Main forward pass
                class_out = model(batch_x, batch_l)
                class_out = pad2list(class_out, batch_l)
                lab = pad2list(lab, batch_l).cuda()
                loss = criterion(class_out, lab)
                class_out_lab = torch.argmax(class_out, dim = 1).cuda()
                accuracy_tensor = torch.where(class_out_lab - lab < .5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                additional_accuracy = torch.sum(accuracy_tensor)
                total_accurate += additional_accuracy.item()
                total_loss += loss.item()
                total_number += lab.shape[0]
                if config.use_gpu:
                    test_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
                else:
                    test_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))
            domain_test_fer = np.mean(test_fer)
            
            accuracies[test_domain_number].append(total_accurate / total_number)
            losses[test_domain_number].append(total_loss / (n + 1))

            print_log += " || Domain: {:d} :: Test: {:.3f} :: Test FER: {:.2f}".format(test_domain_number, losses[test_domain_number][-1], domain_test_fer)

        if len(old_tasks) > 0:
            # Run through old task sample
            total_loss = 0
            total_accurate = 0
            total_number = 0
            test_fer = []
            for (n, (batch_x, batch_l, lab)) in enumerate(old_task):
                if n > debug_cutoff:
                    break

                batch_l = torch.clamp(batch_l, max=config.seq_len)

                _, indices = torch.sort(batch_l, descending=True)
                if config.use_gpu:
                    batch_x = Variable(batch_x[indices]).cuda()
                    batch_l = Variable(batch_l[indices]).cuda()
                    lab = Variable(lab[indices]).cuda()
                else:
                    batch_x = Variable(batch_x[indices])
                    batch_l = Variable(batch_l[indices])
                    lab = Variable(lab[indices])

                optimizer.zero_grad()
                # Main forward pass
                class_out = model(batch_x, batch_l)
                class_out = pad2list(class_out, batch_l)
                lab = pad2list(lab, batch_l).cuda()
                loss = criterion(class_out, lab)
                class_out_lab = torch.argmax(class_out, dim = 1).cuda()
                accuracy_tensor = torch.where(class_out_lab - lab < .5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
                additional_accuracy = torch.sum(accuracy_tensor)
                total_accurate += additional_accuracy.item()
                total_loss += loss.item()
                total_number += lab.shape[0]
                if config.use_gpu:
                    test_fer.append(compute_fer(class_out.cpu().data.numpy(), lab.cpu().data.numpy()))
                else:
                    test_fer.append(compute_fer(class_out.data.numpy(), lab.data.numpy()))
            domain_test_fer = np.mean(test_fer)

            print_log += " || Sample: Test: {:.3f} :: Test FER: {:.2f}".format(total_loss / (n + 1), domain_test_fer)
        logging.info(print_log)

        epoch_i += 1

    return accuracies, losses

# *********************************** Domain 0 *********************************** #
accuracies, losses = train(data_loader_train_domain0, data_loader_dev_domain0, [data_loader_test_domain0])

accuracies_0_0 = accuracies[0]
losses_0_0 = losses[0]
pickle.dump([accuracies_0_0, losses_0_0], outfile_0)
outfile_0.close()

plt.plot(accuracies_0_0)
plt.savefig('accuracies_0_0.png')
plt.clf()

plt.plot(losses_0_0)
plt.savefig('losses_0_0.png')
plt.clf()

# *********************************** Domain 1 *********************************** #
accuracies, losses = train(data_loader_train_domain1, data_loader_dev_domain1, [data_loader_test_domain0, data_loader_test_domain1], [dataset_test_domain0])

accuracies_0_1 = accuracies_0_0 + accuracies[0]
losses_0_1 = losses_0_0 + losses[0]
accuracies_1_1 = accuracies[1]
losses_1_1 = losses[1]
pickle.dump([accuracies_0_1, accuracies_1_1, losses_0_1, losses_1_1], outfile_1)
outfile_1.close()

xs_1 = np.arange(len(accuracies_0_0), len(accuracies_0_0) + len(accuracies[0]))

plt.plot(accuracies_0_1)
plt.plot(xs_1, accuracies_1_1)
plt.savefig('accuracies_1_1.png')
plt.clf()

plt.plot(losses_0_1)
plt.plot(xs_1, losses_1_1)
plt.savefig('losses_1_1.png')
plt.clf()

if fisher_matrices is not None:
    pickle.dump(fisher_matrices, matrix_file)
if fisher_matrix_norms is not None:
    pickle.dump(fisher_matrix_norms, norm_file)
matrix_file.close()
norm_file.close()
