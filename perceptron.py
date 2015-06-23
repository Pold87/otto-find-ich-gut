from __future__ import division

from multilayer_perceptron  import MultilayerPerceptronClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from os import path

def train_clf(clf, train_data):

    # Transform string target values to numerical values 
    lbl_enc = LabelEncoder()
    labels = train_data.target.values
    
    y = lbl_enc.fit_transform(labels)

    # Specify feature columns
    X = train_data.drop(['id', 'target'], axis = 1)

    # Train the classifier
    clf.fit(X, y)

    return clf # See if this could be also done inplace

def make_submission(probs, ids):

    header = ['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']

    df = pd.DataFrame(probs, index = ids, columns = header)
    file_num = 0

    submissions_folder = "submissions/"

    while path.isfile(submissions_folder + 'submission-{}.csv'.format(file_num)):
        file_num += 1

    # Write final submission
    df.to_csv(submissions_folder + 'submission-{}.csv'.format(file_num), index_label = 'id')


def main():
    
    # Load train and test data
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    ids = test_data.id

    # clf = RandomForestClassifier(n_estimators = 10, n_jobs = 32) 
    clf = MultilayerPerceptronClassifier(hidden_layer_sizes = (128, 128, 128), \
                                         max_iter = 10, verbose = True)
    

    clf = train_clf(clf, train_data)

    probs = clf.predict_proba(test_data.drop('id', axis = 1))
    make_submission(probs, ids)
    
if __name__ == "__main__":
    main()

    


