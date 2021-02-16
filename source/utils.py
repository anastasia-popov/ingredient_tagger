import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import pickle
import time
import datetime


def get_attention_masks(input_ids):
    """
    Returns list of attention masks for ids greater than 0
    """
    return [[int(token_id > 0) for token_id in sent] for sent in input_ids]


def get_data_loader(batch_size, training_dir, data_file):
    """
    Builds pytorch dataloader
    """
    print("Get data loader.")

    with open(os.path.join(training_dir, data_file), 'rb') as f:
        train_x, train_y = pickle.load(f)
    
    attention_masks = get_attention_masks(train_x)
        
    train_ds = TensorDataset(torch.tensor(train_x), 
                             torch.ByteTensor(attention_masks), 
                             torch.tensor(train_y))
    train_sampler = RandomSampler(train_ds)


    return torch.utils.data.DataLoader(train_ds,
                                       sampler=train_sampler,
                                       batch_size=batch_size)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))