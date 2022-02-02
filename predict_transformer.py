# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:46:02 2021

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

pretrained_model_name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
data_dir = './data/'
n_splits = 5
n_epochs = 10
max_len = 128
batch_size = 64
dimension = 384

class clit(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.25)
        self.layer_norm = nn.LayerNorm(dimension)
        self.act = nn.ReLU()
        self.linear = nn.Linear(in_features=dimension, out_features=int(dimension/2))
        self.regressor = nn.Linear(in_features=int(dimension/2), out_features=1)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
                
        model_output = self.transformer_model(input_ids, 
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids)[1]
        # model_output = self.head(model_output[1])
        model_output = self.layer_norm(model_output)
        model_output = self.dropout(model_output)
        model_output = self.linear(model_output)
        model_output = self.act(model_output)
        model_output = self.dropout(model_output)
        model_output = self.regressor(model_output)
        
        return model_output

def load_test_data():
    test_df = pd.read_csv(data_dir + 'test.csv')
    test_df.drop(['url_legal', 'license'], axis=1, inplace=True)
    return test_df

def load_predict_save():
    test_df = load_test_data()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokens = tokenizer(test_df['excerpt'].values.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    predictions = np.zeros(len(test_df))
    for i in range(n_splits):
        model = torch.load(f'model_{i}.pth')
        model.eval()
        with torch.no_grad():
            predictions = predictions + model(**tokens)[:,0].detach().numpy()
    predictions = predictions / n_splits
   
    sample_prediction = pd.DataFrame()
    sample_prediction['id'] = test_df['id'].values
    sample_prediction['target'] = predictions
    sample_prediction.to_csv('sample_submission.csv', index=False)
    
if __name__ == '__main__':
    load_predict_save()