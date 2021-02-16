import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle
import time

from model import IngredientTagger
from utils import get_data_loader, format_time


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_params.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IngredientTagger(model_info['num_tags'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print("Done loading model.")
    return model


def train(model, train_dataloader, epochs, optimizer, lr, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_dataloader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    lr           - Learning rate used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    for epoch in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_loss = 0

        model.train()

        for batch in train_dataloader:
            
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()        

            loss = model.loss(input_ids, labels, input_mask)

            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
            
        avg_train_loss = total_loss / len(train_dataloader)            

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
      
    parser.add_argument('--num_tags', type=int, default=8, metavar='NL',
                        help='number of labels (default: 8)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    train_loader = get_data_loader(args.batch_size, args.data_dir, 'train_data.pkl')
    
    model = IngredientTagger(args.num_tags).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train(model, train_loader, args.epochs, optimizer, args.lr, device)

    model_info_path = os.path.join(args.model_dir, 'model_params.pth')

    with open(model_info_path, 'wb') as f:
        model_info = {
            'num_tags': args.num_tags,
        }
        torch.save(model_info, f) 
    
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    
