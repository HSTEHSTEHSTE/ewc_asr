import torch
import pickle

with open("/home/xli257/ASR/train_si284/chunk_0.pt", 'rb') as chunk_0_file:
    chunk_0 = torch.load(chunk_0_file)
