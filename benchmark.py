from __future__ import division
import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


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


def load_train_data(path=None, train_size=0.999):
    path = sys.argv[1] if len(sys.argv) > 1 else path
    if path is None:
        try:
            # Unix
            df = pd.read_csv('data/train.csv')
        except IOError:
            # Windows
            df = pd.read_csv('data\\traifan.csv')
    else:
        df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))


def load_test_data(path=None):
    path = sys.argv[2] if len(sys.argv) > 2 else path
    if path is None:
        try:
            # Unix
            df = pd.read_csv('data/test.csv')
        except IOError:
            # Windows
            df = pd.read_csv('data\\test.csv')
    else:
        df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def train():
    X_train, X_valid, y_train, y_valid = load_train_data()

    
    n_estimators = 5000
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features = 0.7, max_depth = None)
    # clf = GradientBoostingClassifier(n_estimators=n_estimators)
    print(" -- Start training Random Forest Classifier.")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_valid)
    print(" -- Finished training.")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

    return clf, encoder


def make_submission(clf, encoder, path='submissions/my_submissionGradient.csv'):

    path = sys.argv[3] if len(sys.argv) > 3 else path
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
    clf, encoder = train()
    make_submission(clf, encoder)
    print(" - Finished.")


if __name__ == '__main__':
    main()
