from __future__ import division
import sys

import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import grid_search


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


def load_train_data(train_size=0.8, percentage = 0.001):

    df = pd.read_csv('data/train.csv')

    num_samples = int(len(df) * percentage)
    
    sample_rows = random.sample(df.index, num_samples)

    df_sampled = df.ix[sample_rows]
    
    X_train, X_valid, y_train, y_valid = train_test_split(df_sampled.drop(['id', 'target'], axis = 1),
                                                          df_sampled.target, 
                                                          train_size=train_size)
    print('X_train', X_train)
    print('X_valid', X_valid)
    print('y_train', y_train)
    print('y_valid', y_valid)

    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))


def train(clf):
    X_train, X_valid, y_train, y_valid = load_train_data()

    print(" -- Start training.")
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    y_prob = clf.predict_proba(X_valid)

    print(" -- Finished training.")


    print(y_prob)
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    # assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))
    return clf

def main():
    parameters = {'criterion' : ('gini', 'entropy'),
                  'n_estimators' : [500, 700], 
                  'max_depth' : [7, None], 
                  'max_features' : ['log2', 'sqrt'], 
                  'min_samples_leaf' : [1, 2]}

    clf_cv = RandomForestClassifier(n_jobs = 40)

    clf1 = grid_search.GridSearchCV(clf_cv, parameters)

    classifiers = [clf1]

    for i, clf in enumerate(classifiers):
        clf_fitted = train(clf)


if __name__ == '__main__':
    main()





