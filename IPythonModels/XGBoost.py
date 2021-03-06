
# coding: utf-8

# In[1]:

from __future__ import division
import sys

import xgboost as xgb

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


# In[2]:

np.random.seed(83415)

encoder = LabelEncoder()

def load_train_data(path='../data/train80.csv'):
    df = pd.read_csv(path)
    X_train = df.iloc[:, 1:-1]
    y_train = df.target
    y_train_enc = encoder.fit_transform(y_train)
    return X_train.astype(float), y_train_enc.astype(float)


def load_test_data(path="../data/holdout20.csv"):
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


# In[3]:

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


# In[ ]:

space = {

    "set.seed": 42,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "nthread": 2,
    "silent": 1,

    "max_depth": hp.quniform('depth', 8, 12, 1),
    "eta": 0.07,
    "subsample": hp.uniform('subsample', 0.7, 1),
    "colsample_bytree": hp.uniform('colsample', 0.7, 1),
    "gamma": hp.uniform('gamma', 0.0, 1),
    "min_child_weight": hp.quniform('childweight', 3, 10, 1),
    "max_delta_step": hp.quniform('delta', 0, 3, 1)

}


def objective(hyperparameter):
    num_round = 800
    bst = xgb.train(hyperparameter, xg_train, num_round)
    ypred = bst.predict(xg_valid)
    save(ypred)
    sub = pd.read_csv("xgboostpython.csv").iloc[:, 1:]
    ll = logloss_mc(y_valid, sub)
    print "Log loss is:", ll
    print(hyperparameter)
    return {'loss': ll,
            'status': STATUS_OK}


trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=150,
            trials=trials)

print(best)


# In[10]:

print trials.losses()

print(best)