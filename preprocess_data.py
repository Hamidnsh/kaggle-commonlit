# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:24:50 2021

@author: snourashrafeddin
"""

import pandas as pd 
import numpy as np 
import pickle
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model, load_model
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from flaml import AutoML


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()



data_dir = './data/'
model_names = ['paraphrase-MiniLM-L6-v2', 'paraphrase-MiniLM-L12-v2']
number_of_splits = 5

def load_data():
    train_df = pd.read_csv(data_dir + 'train.csv')
    train_df.drop(['url_legal', 'license', 'standard_error'], axis=1, inplace=True)
    # test_df = pd.read_csv(data_dir + 'test.csv')
    # test_df.drop(['url_legal', 'license'], axis=1, inplace=True)
    return train_df

# def create_head_nn(input_dim):
#     model = Sequential()
#     model.add(Dense(64, input_dim=input_dim, activation='relu'))
#     model.add(Dropout(0.75))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dropout(0.75))
#     model.add(Dense(1, activation='relu'))
#     model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9), metrics='mse') #SGD(lr=0.01, momentum=0.9)
#     return model

def create_head_regression():
    return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1, epsilon=0.1))
    # return AutoML()
    
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower()) if w not in stopwords.words('english')])

def train_regression_model(train_df):
    regression_models = {}
    train_df['excerpt_lemm'] = train_df['excerpt'].apply(lemmatize_text)
    mse_sum = 0
    for model_name in model_names:
        print(model_name + ' ...')
        regression_models[model_name] = []
        model = SentenceTransformer(model_name)
        sentences = train_df['excerpt_lemm'].values
        # sentences = train_df['excerpt'].values
        sentence_embeddings = model.encode(sentences)
        targets = train_df.target.values
        kf = KFold(n_splits=number_of_splits, shuffle=True)
        for i in range(number_of_splits):
            print('fold ' + str(i))
            fold = next(kf.split(sentence_embeddings))
            fit_mat = sentence_embeddings[fold[0], :]
            val_mat = sentence_embeddings[fold[1], :]
            fit_target = targets[fold[0]]
            val_target = targets[fold[1]]
            
            head_model = create_head_regression()
            head_model.fit(fit_mat, fit_target)
            predictions = head_model.predict(val_mat)
            mse = np.sqrt(np.mean((predictions - val_target)**2))
            print('rmse: ' + str(mse))
            mse_sum = mse_sum + mse
            
            regression_models[model_name].append(head_model)
    print('avg mse' + str(mse_sum/(len(model_names)*number_of_splits)))
    return regression_models
        
    
def infer_from_model():
    test_df = pd.read_csv(data_dir + 'test.csv')
    test_df.drop(['url_legal', 'license'], axis=1, inplace=True)
    test_df['excerpt_lemm'] = test_df['excerpt'].apply(lemmatize_text)
    sample_prediction = pd.DataFrame()
    sample_prediction['id'] = test_df['id'].values
    sample_prediction['target'] = 0
    number_of_head_models = 0
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        sentences = test_df['excerpt_lemm'].values
        sentence_embeddings = model.encode(sentences)
        for head_model in regression_models[model_name]:
            predictions = head_model.predict(sentence_embeddings)
            sample_prediction['target'] = sample_prediction['target'] + predictions
            number_of_head_models = number_of_head_models + 1
    
    sample_prediction['target'] =  sample_prediction['target'] / number_of_head_models
    sample_prediction.to_csv('sample_submission.csv', index=False)

def load_reg_models():
    with open('regression_models.pkl', 'rb') as fin:
       regression_models = pickle.load(fin)
   


def save_reg_models(regression_models):
    with open('regression_models.pkl', 'wb') as fout:
        pickle.dump(regression_models, fout)
    
        
if __name__ == '__main__':
    train_df = load_data()
    regression_models = train_regression_model(train_df)
    save_reg_models(regression_models)
    