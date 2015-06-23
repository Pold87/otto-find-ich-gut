
# coding: utf-8

# In[5]:

from __future__ import division
import sys

import xgboost as xgb

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


# In[6]:

np.random.seed(83415)

encoder = LabelEncoder()

def load_train_data(path='../data/train80varmean.csv'):
    df = pd.read_csv(path)
    X_train = df.iloc[:, 1:-1]
    y_train = df.target
    y_train_enc = encoder.fit_transform(y_train)
    return X_train.astype(float), y_train_enc.astype(float)


def load_test_data(path="../data/holdout20varmean.csv"):
    df = pd.read_csv(path)
    
    X_test, ids, y_valid = df.iloc[:, 1:-1], df.id, df.target
    
    y_valid_enc = encoder.fit_transform(y_valid)

    return X_test.astype(float), ids.astype(str), y_valid_enc.astype(float), y_valid.astype(str)


def save(ypred):
    df = pd.DataFrame(ypred, columns=['Class_{}'.format(i)
                                       for i in range(1, 10)],
                      index=np.arange(len(ypred)))
    outfile = "xgboostpython.csv"
    df.to_csv(outfile, header=True, index_label='id')

    
X_train, y_train = load_train_data()
X_valid, ids, y_valid_enc, y_valid = load_test_data()

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_valid = xgb.DMatrix(X_valid, label=y_valid_enc)


# In[7]:

sample_sub = "../submissions/sampleSubmission.csv"
sample_sub_df = pd.read_csv(sample_sub)

def normalize(row, epsilon=1e-15):
    
    row = row / np.sum(row)
    row = np.maximum(epsilon, row)
    row = np.minimum(1 - epsilon, row)
    
    return row
    
def logloss_mc(y_true, y_probs):
    
    # Normalize probability data frame
    y_probs = y_probs.apply(normalize, axis=1)
        
    log_vals = []
        
    for i, y in enumerate(y_true):
        c = int(y.split("_")[1])
        log_vals.append(- np.log(y_probs.iloc[i,c - 1]))
        
    return np.mean(log_vals)


# In[14]:

hyperparameters = {

    "set.seed": 42,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "nthread": 2,
    "silent": 1,

    "max_depth": 12,
    "eta": 0.01,
    "subsample": 0.86,
    "colsample_bytree": 0.8,
    "gamma": 0.73,
    "min_child_weight": 6,
    "max_delta_step": 1

}


# In[ ]:

num_round = 100
bst = xgb.train(hyperparameters, xg_train, num_round)
ypred = bst.predict(xg_valid)
save(ypred)
sub = pd.read_csv("xgboostpython.csv").iloc[:, 1:]
ll = logloss_mc(y_valid, sub)


# In[13]:

print ll

