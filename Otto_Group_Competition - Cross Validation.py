
# coding: utf-8
# Otto Group Product Classification Challenge using nolearn/lasagne
# This short notebook is meant to help you getting started with nolearn and lasagne in order to train a neural net and make a submission to the Otto Group Product Classification Challenge.
# 
# * [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
# * [Get the notebook from the Otto Group repository](https://github.com/ottogroup)
# * [Nolearn repository](https://github.com/dnouri/nolearn)
# * [Lasagne repository](https://github.com/benanne/Lasagne)
# * [A nolearn/lasagne tutorial for convolutional nets](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)

# ## Imports

# In[30]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import theano


# In[163]:

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from sklearn.cross_validation import train_test_split
import random


# # Cross validation

# ## Log loss

# In[336]:

sample_sub = "submissions/sampleSubmission.csv"
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
        


# ## Load data

# In[189]:

def load_train_data_non_lasagne(df, train_size=0.8, percentage=1, standardize=False):

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
    
    X_train, X_valid, y_train, y_valid = train_test_split(df_sampled.drop(['id', 'target'], axis=1),
                                                          df_sampled.target, 
                                                          train_size=train_size)
    
    return (X_train, X_valid,
            y_train.astype(str), y_valid.astype(str))


# ## Utility functions

# In[182]:

## Get random rows


# In[193]:

df = pd.read_csv("data/train.csv")


# In[262]:

X_train, X_valid, y_train, y_valid = load_train_data_non_lasagne(pd.read_csv("data/train.csv"))


# In[265]:

len(y_valid)


# In[266]:

def load_train_data_cross_validation(X_train, y_train):
    
    X, labels = X_train.astype(np.float32), y_train
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler


# In[274]:

def load_test_data_cross_validation(X_valid, scaler):
    X_valid, ids = X_valid.astype(np.float32), np.arange(1, len(y_valid) + 1).astype(str)
    X_valid = scaler.transform(X_valid)
    return X_valid, ids


# In[198]:

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


# In[199]:

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids


# In[200]:

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
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


# In[201]:

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


# # Perform cross validation

# In[ ]:

# In[275]:

X, y, encoder, scaler = load_train_data_cross_validation(X_train, y_train)


num_classes = len(encoder.classes_)
num_features = X.shape[1]



# In[355]:

X_test, ids = load_test_data_cross_validation(X_valid, scaler)


# ## Train Neural Net

# In[278]:

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]


# In[279]:

net2 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=300,
                 dropout_p=0.5,
                 dense1_num_units=300,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 
                 # optimization method:
                 update_learning_rate=theano.shared(float32(0.05)),
                 update_momentum=theano.shared(float32(0.8)),


                on_epoch_finished=[
                    AdjustVariable('update_learning_rate', start=0.05, stop=0.00001),
                    AdjustVariable('update_momentum', start=0.8, stop=0.9999),
                    EarlyStopping(patience=200),
        ],
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=10)


# In[280]:

net2.fit(X, y)


# ## Prepare Submission File

# In[281]:

make_submission(net2, X_test, ids, encoder, "cross_validation1.csv")


# In[350]:

y_probs = pd.read_csv("cross_validation1.csv").iloc[:,1:]


# In[351]:

y_true = y_valid


# In[354]:

logloss_mc(y_true, y_probs)


# # Polish submission file

# In[341]:

def polish(row, threshold=0.01):
    
    for i, x in enumerate(row):
        if x < threshold:
            row[i] = 0
            
    return row    


# In[343]:

# y_probs = y_probs.apply(polish, axis=1)


# # Gradient Descent

# In[359]:

import graphlab as gl


# In[361]:

# Load the data

graph_df = pd.DataFrame(X)
graph_df['target'] = y

train = gl.SFrame(graph_df)
test = gl.SFrame(X_test)
sample = gl.SFrame.read_csv('submissions/sampleSubmission.csv')

# Train a model
m = gl.boosted_trees_classifier.create(dataset = train,
                                       target='target',
                                       max_iterations=100,
                                       max_depth = 10,
                                       row_subsample = 0.86,
                                       column_subsample = 0.78,
                                       min_loss_reduction = 1.05,
                                       min_child_weight = 4,
                                       validation_set = None)
 
# Make submission
preds = m.predict_topk(test, output_type='probability', k=9)
preds['id'] = preds['id'].astype(int) + 1
preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
preds = preds.sort('id')
 
assert sample.num_rows() == preds.num_rows()

preds.save("graphlab_crazy_cross.csv", format = 'csv')


# In[ ]:



