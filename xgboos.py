from __future__ import division
import sys

import xgboost as xgb

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

np.random.seed(83415)



def load_train_data(path='data/train80.csv'):
    df = pd.read_csv(path)
    X_train = df.iloc[:, 1:-1]
    y_train = df.target
    return X_train, y_train


def load_test_data(path="data/holdout20.csv"):
    df = pd.read_csv(path)
    X_test, ids, y_valid = df[:, 1:-1], df[:, 0], df.target

    return X_test.astype(float), ids.astype(str), y_valid.astype(str)


space = {

    "set.seed": 42,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 9,
    "nthread": 2,
    "silent": 1,

    "max_depth": hp.quniform('depth', 8, 12, 1),
    "eta": hp.uniform('eta', 0.001, 0.2),
    "subsample": hp.uniform('subsample', 0.7, 1),
    "colsample_bytree": hp.uniform('colsample', 0.7, 1),
    "gamma": hp.uniform('gamma', 0.0, 1),
    "min_child_weight": hp.quniform('childweight', 3, 10, 1),
    "max_delta_step": hp.quniform('delta', 0, 3, 1)

}

X_train, y_train = load_train_data()
X_valid, ids, y_valid = load_test_data()

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_valid = xgb.DMatrix(X_valid, label=y_valid)


def save(ypred):
    df = pd.DataFrame(y_pred, columns=['Class_{}'.format(i)
                                       for i in range(1, 10)],
                      index=np.arange(len(ypred)))
    outfile = "xgboostpython.csv"
    df.to_csv(outfile, header=True, index_label='id')


def objective(hyperparameter):
    num_round = 10
    bst = xgb.train(hyperparameter, xg_train, num_round)
    ypred = bst.predict(X_test)
    save(ypred)
    sub = pd.read_csv("xgboostpython.csv").iloc[:, 1:]
    ll = logloss_mc(y_valid, sub)
    print(ll)
    return {'loss': ll,
            'status': STATUS_OK}


trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=2,
            trials=trials)

print(best)


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss

    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

# TODO: maybe remove this methods
def train():

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

    return clf, encoder


def main():
    print(" - Start.")
    clf, encoder = train()
    make_submission(clf, encoder)
    print(" - Finished.")


#if __name__ == '__main__':
#    main()
