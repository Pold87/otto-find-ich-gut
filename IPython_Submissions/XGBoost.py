from __future__ import division


import xgboost as xgb

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


np.random.seed(83415)

encoder = LabelEncoder()


def load_train_data(path='../data/train.csv'):
    df = pd.read_csv(path)
    X_train = df.iloc[:, 1:-1]
    y_train = df.target
    y_train_enc = encoder.fit_transform(y_train)
    return X_train.astype(float), y_train_enc.astype(float)


def load_test_data(path="../data/test.csv"):
    df = pd.read_csv(path)

    X_test, ids = df.iloc[:, 1:], df.id

    return X_test.astype(float), ids.astype(str)


def save(ypred):
    df = pd.DataFrame(ypred, columns=['Class_{}'.format(i)
                                      for i in range(1, 10)],
                      index=np.arange(len(ypred)))
    outfile = "xgboostpython_full_submission_2.csv"
    df.to_csv(outfile, header=True, index_label='id')


X_train, y_train = load_train_data()
X_valid, ids = load_test_data()

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_valid = xgb.DMatrix(X_valid)

sample_sub = "../submissions/sampleSubmission.csv"
sample_sub_df = pd.read_csv(sample_sub)

space = {

    "set.seed": 3814,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "nthread": 2,
    "silent": 0,

    "max_depth": 12,
    "eta": 0.04,
    "subsample": 0.922,
    "colsample_bytree": 0.814,
    "gamma": 0.97,
    "min_child_weight": 3,
    "max_delta_step": 1

}


num_round = 1500
bst = xgb.train(space, xg_train, num_round)
ypred = bst.predict(xg_valid)
save(ypred)
