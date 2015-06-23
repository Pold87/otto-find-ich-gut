
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import theano

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


# # Utility Functions

# In[2]:

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler


# In[3]:

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    df = df.iloc[:,:-1]
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids


# # Load Data

# In[5]:

X, y, encoder, scaler = load_train_data("../data/train80.csv")
X_test, ids = load_test_data('../data/holdout20.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]


# # Adjust network parameters over time

# In[7]:

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


# # Train Neural Net

# In[11]:

layers0 = [('input', InputLayer),
           ('dropoutin', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]


# In[20]:

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dropoutin_p = 0.0002,
                 
                 dense0_num_units=240,
                 dropout0_p=0.235,
                 
                 dense1_num_units=510,
                 dropout1_p=0.29,
                 
                 dense2_num_units=480,
                 
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,                 
                 
                 update_learning_rate=theano.shared(float32(0.03)),
                 update_momentum=theano.shared(float32(0.92)),

                on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
                    AdjustVariable('update_momentum', start=0.92, stop=0.98),
        ],
                 
                 eval_size=None,
                 verbose=1,
                 max_epochs=120)


# In[ ]:

net0.fit(X, y)


# # Log loss

# In[14]:

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

df_holdout = pd.read_csv("../data/holdout20.csv")
y_valid = df_holdout.target


# In[15]:

def make_submission_fix(clf, X_test, ids, encoder, name='localcrosslasagne.csv'):
    y_prob = clf.predict_proba(X_test)
    
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
            
    print("Wrote submission to file {}.".format(name))


# In[18]:

make_submission_fix(net0, X_test, ids, encoder)
sub = pd.read_csv("localcrosslasagne.csv").iloc[:, 1:]
ll = logloss_mc(y_valid, sub)


# In[19]:

print ll

