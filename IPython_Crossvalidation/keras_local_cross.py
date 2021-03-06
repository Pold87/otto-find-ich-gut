
# coding: utf-8

# In[4]:

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4 
    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881 
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

np.random.seed(83415)

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:-1].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


# In[5]:

print("Loading data...")
X, labels = load_data('../data/train80.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('../data/holdout20.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')


# In[ ]:

print("Building model...")

model = Sequential()
model.add(Dense(dims, 240, init='glorot_uniform'))
model.add(PReLU((240,)))
model.add(BatchNormalization((240,)))
model.add(Dropout(0.2))

model.add(Dense(240, 510, init='glorot_uniform'))
model.add(PReLU((510,)))
model.add(BatchNormalization((510,)))
model.add(Dropout(0.25))

model.add(Dense(510, 480, init='glorot_uniform'))
model.add(PReLU((480,)))
model.add(BatchNormalization((480,)))
model.add(Dropout(0.4))

model.add(Dense(480, nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam")

print("Training model...")

model.fit(X, y, nb_epoch=50, batch_size=16, validation_split=0)

print("Generating submission...")

proba = model.predict_proba(X_test)
make_submission(proba, ids, encoder, fname='keras-otto.csv')

