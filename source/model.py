import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertForTokenClassification
from torchcrf import CRF

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

    
class IngredientTagger(nn.Module):
    """
    Main model for ingredient tagging
    """
    def __init__(self, num_tags):
        """
        Model initialization

        Parameters:
            num_tags (int) - total number of tags
        """
        super(IngredientTagger, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            "bert-base-uncased", 
             num_labels = num_tags, 
             output_attentions = False, 
             output_hidden_states = False,
         )
        
    
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    def loss(self, x, y, attention_mask):
        """
        Return the loss after one forward pass

        Parameters:
            x (LongTensor) - tensor of tokenized and padded sentences
            y (LongTensor) - tensor of tokenized and padded tags
            attention_mask(ByteTensor) - tensor of attention_masks

        Returns:
            loss - loss after one model forward pass  

        """
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        encoded_layers = self.bert(x,
                                   labels=y,
                                   attention_mask=attention_mask,
                                   token_type_ids=None)
        bert_loss  = encoded_layers.loss            
        
        return bert_loss
        
        
    def forward(self, x, attention_mask):
        """
        Model's forward pass (used for prediction)
        """
        x = x.to(self.device) 
       
        self.bert.eval()
        with torch.no_grad():
            encoded_layers = self.bert(x,
                                    attention_mask=attention_mask,
                                       token_type_ids=None)
                        
            logits = encoded_layers.logits  
            y_hat = logits.argmax(-1) 
    
        return y_hat
        