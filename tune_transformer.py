# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:15:27 2021

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


torch.manual_seed(2021)
pretrained_model_name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
# pretrained_model_name = "roberta-base"
data_dir = './data/'
n_splits = 5
n_epochs = 10
max_len = 128
batch_size = 64
dimension = 384

class custom_dataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.exceprts = self.data['excerpt'].values.tolist()
        self.targets = self.data['target'].values.tolist()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.exceprts[idx]
        target = self.targets[idx]
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len)
        return {
            'input_ids':torch.tensor(tokens['input_ids'], dtype=torch.long),
            'token_type_ids':torch.tensor(tokens['token_type_ids'], dtype=torch.long),
            'attention_mask':torch.tensor(tokens['attention_mask'], dtype=torch.long),
            'target':torch.tensor(target, dtype=torch.float)}
        

def load_train_data():
    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df.drop(['url_legal', 'license', 'standard_error'], axis=1, inplace=True)
    train_df["kfold"] = -1
    kf = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=train_df)):
        train_df.loc[v_, 'kfold'] = f
    return train_df


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

def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = clit().to(device)        
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
    return model, loss_fn, optimizer, device, scheduler

def create_dataloader(train_df, tokenizer, max_len, batch_size, fold=0):
    train_set, val_set = train_df.loc[train_df['kfold'] != fold],  train_df[train_df['kfold'] == fold]
    train_dataset = custom_dataset(train_set, tokenizer, max_len)
    val_dataset = custom_dataset(val_set, tokenizer, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader


def train_model(train_dataloader, model, loss_fn, device, optimizer, scheduler):
    model.train()
    train_loss = 0
    size = 0
    for batch, data in enumerate(train_dataloader):
        input_ids = data['input_ids'].to(device)
        token_type_ids = None
        token_type_ids = data['token_type_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        y = data['target'].to(device)
        
        pred = model(input_ids, token_type_ids, attention_mask)[:, 0]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += np.sum((pred.detach().numpy() - y.detach().numpy())**2)
        size += len(y)
        # if batch % 5 == 0:
        #     loss = loss.item()
        #     print(f"loss : {loss:>7f} ...")
    scheduler.step()
    train_rmse = np.sqrt(train_loss/size)
    print(f'train rmse: {train_rmse} ')
    return train_rmse

def evaluate_model(val_dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    size = 0
    with torch.no_grad():
        for batch, data in enumerate(val_dataloader):
            input_ids = data['input_ids'].to(device)
            token_type_ids = None
            token_type_ids = data['token_type_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            y = data['target'].to(device)
            
            pred = model(input_ids, token_type_ids, attention_mask)[:, 0]
            test_loss += np.sum((pred.numpy() - y.numpy())**2)
            size += len(y)
        test_rmse = np.sqrt(test_loss/size)
        print(f'test rmse: {test_rmse}')
        return test_rmse
             
    
def fit_models(train_df, max_len, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    for fold in range(n_splits):
        print(f'This is fold {fold} ....................')
        train_dataloader, val_dataloader = create_dataloader(train_df, tokenizer, max_len, batch_size, fold=fold)
        model, loss_fn, optimizer, device, scheduler = create_model()
        loss_df = pd.DataFrame()
        loss_df['epoch'] = list(range(n_epochs))
        loss_df['train_loss'] = [np.nan]*n_epochs
        loss_df['test_loss'] = [np.nan]*n_epochs
        for _epoch in range(n_epochs):
            print(f'This is epoch {_epoch} ...')
            tr_loss = train_model(train_dataloader, model, loss_fn, device, optimizer, scheduler)
            ts_loss = evaluate_model(val_dataloader, model, loss_fn, device)
            loss_df.loc[loss_df['epoch'] == _epoch, 'train_loss'] = tr_loss
            loss_df.loc[loss_df['epoch'] == _epoch, 'test_loss'] = ts_loss
        torch.save(model, f'model_{fold}.pth')
        loss_df.to_csv(f'model_loss_{fold}.csv', index=False)

if __name__ == '__main__':
    train_df = load_train_data()
    fit_models(train_df, max_len, batch_size)
    