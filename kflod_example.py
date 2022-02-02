# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:32:27 2021

@author: snourashrafeddin
"""

import pandas as pd
import numpy as np

batch_size = 64
pretrained_model_name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
data_dir = './data/'
number_of_splits = 5
n_epochs = 5

from sklearn import model_selection
def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    return data

train_df = pd.read_csv(data_dir + 'train.csv')    
train = create_folds(train_df, num_splits=5)