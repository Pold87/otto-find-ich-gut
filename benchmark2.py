
from __future__ import division
import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn import grid_search

np.random.seed(17411)


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


def load_train_data(train_size=0.95, percentage =):
    df = pd.read_csv('data/train.csv')
    X = df.values.copy()
    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))


def train(clf):
    X_train, X_valid, y_train, y_valid = load_train_data()

    print(" -- Start training.")
    clf.fit(X_train, y_train)
    print(clf.best_params_)

    if isClassification:
        y_prob = clf.predict_proba(X_valid)

    else:
        y_prob = clf.predict_proba(X_valid)

    print(" -- Finished training.")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

    return clf, encoder


def make_submission(clf, encoder, i):

    path = 'submissions/' + str(i) + '.csv' 
    
    X_test, ids = load_test_data()
    y_prob = clf.predict_proba(X_test)
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))





def main():
    print(" - Start.")

#    clf1 = svm.SVC(probability = True)
    parameters = {'criterion' : ('gini', 'entropy'),
                  'n_estimators' : [1000, 2000, 3000], 
                  'max_depth' : [6, 7, 8, None], 
                  'max_features' : ['log2', 'sqrt'], 
                  'min_samples_leaf' : [1, 2, 3]}

    parameters = { 'class_weight' : ['auto', None] }

#    clf_cv = RandomForestClassifier(n_jobs = 40)

#     clf1 = grid_search.GridSearchCV(clf_cv, parameters)
    clf1 = svm.SVC(class_weight = 'auto')

 #   clf3 = GradientBoostingClassifier(n_estimators = 200)

    classifiers = [clf1]
  #                 clf2,
#                   clf3]
    
    

    for i, clf in enumerate(classifiers):
        clf_fitted, encoder = train(clf, True)
        make_submission(clf_fitted, encoder, i)


    print(" - Finished.")


if __name__ == '__main__':
    main()
