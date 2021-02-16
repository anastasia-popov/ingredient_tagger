import os
import numpy as np
import torch
from six import BytesIO
import pickle
import json
import torch
from keras.preprocessing.sequence import pad_sequences

from model import IngredientTagger
from utils import get_attention_masks


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


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/json':
        data = list(json.loads(serialized_input_data.decode('utf-8')))
        return data 
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

    
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return json.dumps(prediction_output.tolist())


def predict_fn(input_data,  model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 
                           'tokenizer', 
                           'bert-base-uncased') 
    
    encoded_sents = [tokenizer.encode(sent,add_special_tokens = True) 
                     for sent in input_data]
    padded_sents = pad_sequences(encoded_sents, maxlen=30, dtype="long", 
                          value=0, truncating="post", padding="post")
    
    data = torch.tensor(padded_sents).long().to(device)
    
    attention_masks = torch.ByteTensor(get_attention_masks(data)).to(device)

    model.eval()

    out = model(data, attention_masks)
    
    out_np = out.cpu().detach().numpy()

    return out_np