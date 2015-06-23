from __future__ import division
import sys

import ModelStacking

import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn import grid_search
from sklearn.metrics import accuracy_score

standardize = True


def log_normalize(x):
    return np.log(x + 1)

    
def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss

    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    """

    print(y_true)

    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_train_data(train_size=0.8, percentage=1):

    df = pd.read_csv('data/train.csv')

    if standardize:
        X = df.drop(['id', 'target'], axis=1).apply(func=log_normalize, axis=1)
        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X)
        X.loc[:, 'id'] = df.loc[:, 'id']
        X.loc[:, 'target'] = df.loc[:, 'target']
        df = X
        
    num_samples = int(len(df) * percentage)
    
    sample_rows = random.sample(df.index, num_samples)

    df_sampled = df.ix[sample_rows]
    
    X_train, X_valid, y_train, y_valid = train_test_split(df_sampled.drop(['id', 'target'], axis = 1),
                                                          df_sampled.target, 
                                                          train_size=train_size)

    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))


def train(clf):
    X_train, X_valid, y_train, y_valid = load_train_data()

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_valid)

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    # assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print("Multiclass score " + str(score))
    return clf

def main():
    parameters = {'criterion' : ('gini', 'entropy'),
                  'n_estimators' : [500, 700], 
                  'max_depth' : [7, None], 
                  'max_features' : ['log2', 'sqrt'], 
                  'min_samples_leaf' : [1, 2]}

    
#    clf_1 = RandomForestClassifier(n_estimators = 2000)
#    clf_2 = RandomForestClassifier(n_estimators = 3000)
#    clf_3 = RandomForestClassifier(n_estimators = 5000, max_depth = 17)
#    clf_4 = RandomForestClassifier(n_estimators = 7000, max_depth = 17)    
#    clf_5 = RandomForestClassifier(n_estimators = 2000, min_samples_leaf = 2)
#    clf_6 = RandomForestClassifier(n_estimators = 2000, min_samples_leaf = 1)
#    clf_7 = RandomForestClassifier(n_estimators = 1300)
#    clf_8 = RandomForestClassifier(n_estimators = 1500)
    clf_9 = GradientBoostingClassifier(n_estimators = 300, max_depth = 20, subsample = 0.6, warm_start = True, learning_rate = 0.01)   
#    clf_10 = GradientBoostingClassifier(n_estimators = 100, max_depth = 15, subsample = 0.7, warm_start = False)   
#    
    # clf1 = grid_search.GridSearchCV(clf_cv, parameters)
    classifiers = [clf_9]
    for i, clf in enumerate(classifiers):
        print(clf)
        clf_fitted = train(clf)

if __name__ == '__main__':
    main()
    