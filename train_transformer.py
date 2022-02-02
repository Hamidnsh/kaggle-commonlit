# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:44:39 2021

@author: snourashrafeddin
"""

from sentence_transformers import models, SentenceTransformer
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch import nn
import torch
import numpy as np
import pandas as pd


number_of_splits = 3
data_dir = './data/'

# pretrained_model_names = ["sentence-transformers/paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L12-v2"]
pretrained_model_names = ["sentence-transformers/paraphrase-MiniLM-L12-v2"]

def load_data():
    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df.drop(['url_legal', 'license', 'standard_error'], axis=1, inplace=True)
    input_examples = train_df.apply(lambda x: InputExample(texts=[x.excerpt], label=x.target), axis=1, result_type='reduce')
    train_df['Input_Example'] = input_examples.values
    return train_df


class meanSquaredLoss(nn.Module):
    def __init__(self, model):
        super(meanSquaredLoss, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()
        
    def forward(self, text, true_val):
        pred_val = self.model(text[0])['sentence_embedding']
        return self.loss_fct(pred_val[:,0], true_val)
        
def create_model(pretrained_model_name):
    
    word_embedding_model = models.Transformer(pretrained_model_name)#, 
                                              # max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode='cls')
    
    # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
    #                             out_features=16, 
    #                             activation_function=nn.Tanh())
       
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                                out_features=1, 
                                activation_function=nn.Softshrink(1e-8))
    # linear_model = nn.Linear(in_features=pooling_model.get_sentence_embedding_dimension(),
    #                          out_features=1,
    #                          bias=True)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    return model

def train_model(model, sentences, fold_tag):
    train_loss = meanSquaredLoss(model)
    train_dataloader = DataLoader(sentences, shuffle=True, batch_size=16)
    model.fit(train_objectives=[(train_dataloader, train_loss)],   ### add more configurations 
              epochs=5,
              output_path='trainedmodel' + fold_tag)
    model = SentenceTransformer('trainedmodel' + fold_tag)

    return model    

def create_cv_models(train_df):
    mse_sum = 0
    j = 0
    for model_name in pretrained_model_names:
        print(model_name + ' ...')
        kf = KFold(n_splits=number_of_splits, shuffle=True)
        input_examples = train_df['Input_Example'].values
        targets = train_df['target'].values
        for i in range(number_of_splits):
            print('fold' + str(i))
            fold = next(kf.split(input_examples))
            fit_sentences = input_examples[fold[0]]
            # val_sentences = input_examples[fold[1]]
            # fit_target = targets[fold[0]]
            val_target = targets[fold[1]]
            
            model = create_model(model_name)
            model = train_model(model, fit_sentences, 'fold' + str(j))
            j = j + 1
                      
            predictions = model.encode(train_df['excerpt'].values[fold[1]])

            mse = np.sqrt(np.mean((predictions[:,0] - val_target)**2))
            print('rmse: ' + str(mse))
            mse_sum = mse_sum + mse
            
            
    print('avg mse ' + str(mse_sum/(len(pretrained_model_names)*number_of_splits)))


if __name__ == '__main__':
    print('This is a run with L12, cls instead of mean, 5 epochs, no truncation...')
    train_df = load_data()
    create_cv_models(train_df)
    
# sentences =  [InputExample(texts=['This framework generates embeddings for each input sentence'], label=0.8),
#               InputExample(texts=['Sentences are passed as a list of string.'], label=0.2),
#               InputExample(texts=['The quick brown fox jumps over the lazy dog.'], label=0.1)]




# sentences =  ['This framework generates embeddings for each input sentence',
#                           'Sentences are passed as a list of string.',
#                           'The quick brown fox jumps over the lazy dog.']

