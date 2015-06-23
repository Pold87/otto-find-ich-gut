
# coding: utf-8

# # Otto Group Product Classification Challenge using nolearn/lasagne

# This short notebook is meant to help you getting started with nolearn and lasagne in order to train a neural net and make a submission to the Otto Group Product Classification Challenge.
# 
# * [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
# * [Get the notebook from the Otto Group repository](https://github.com/ottogroup)
# * [Nolearn repository](https://github.com/dnouri/nolearn)
# * [Lasagne repository](https://github.com/benanne/Lasagne)
# * [A nolearn/lasagne tutorial for convolutional nets](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)

# ## Imports

# In[1]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import theano


# In[2]:

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, leaky_rectify, LeakyRectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import functools


# ## Utility functions

# In[3]:

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


# In[4]:

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    df = df.iloc[:,:-1]
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids


# In[5]:

def make_submission(clf, X_test, ids, encoder):
    y_prob = clf.predict_proba(X_test)
    
    i = 0
    while os.path.exists(os.path.join("submissions", "nn-" + str(i) + ".csv")):
        i += 1
    name = os.path.join("submissions", "nn-" + str(i) + ".csv")
    
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
            
    print("Wrote submission to file {}.".format(name))


# ## Load Data

# In[6]:

X, y, encoder, scaler = load_train_data("../data/train80.csv")


# In[7]:

X_test, ids = load_test_data('../data/holdout20.csv', scaler)


# In[8]:

num_classes = len(encoder.classes_)
num_features = X.shape[1]


# # Adjust network parameters over time

# In[9]:

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


 
"""
Preliminary implementation of batch normalization for Lasagne.
Does not include a way to properly compute the normalization factors over the
full training set for testing, but can be used as a drop-in for training and
validation.

Author: Jan Schlüter
"""
 
import lasagne
import theano.tensor as T
 
class BatchNormLayer(lasagne.layers.Layer):
 
    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
            nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).
        
        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = theano.shared(np.zeros(shape, dtype=dtype), 'mean')
        self.std = theano.shared(np.ones(shape, dtype=dtype), 'std')
        self.beta = theano.shared(np.zeros(shape, dtype=dtype), 'beta')
        self.gamma = theano.shared(np.ones(shape, dtype=dtype), 'gamma')
 
    def get_params(self):
        return [self.gamma] + self.get_bias_params()
 
    def get_bias_params(self):
        return [self.beta]
 
    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)
 
def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).

    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)

# ## Train Neural Net

# In[12]:

layers0 = [('input', InputLayer),
           ('dropoutin', DropoutLayer),
           ('dense0', DenseLayer),
           ('batch0', BatchNormLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
		   ('batch1', BatchNormLayer),
           ('dropout1', DropoutLayer),
           #('dense2', DenseLayer),
		   #('batch2', BatchNormLayer),
           #('dropout2', DropoutLayer),
           ('output', DenseLayer)]


# # Log loss

# In[10]:

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


# # Define hyperparameter

# In[13]:

def f21(f):
    return theano.shared(float32(f))

space = {  'layers' : layers0,
                'input_shape' : (None, num_features),
                
                'dropoutin_p' : hp.uniform('dropin', 0, 0.1),
                
                'dense0_num_units': hp.quniform('dense0', 100, 1000, 20),
                'dense0_nonlinearity' : hp.uniform('leaky0', 0, 1),
                'dropout0_p': hp.uniform('drop0', 0, 0.5),
                
                'dense1_num_units' : hp.quniform('dense1', 100, 1000, 20),
                'dense1_nonlinearity' : hp.uniform('leaky1', 0, 1),
                'dropout1_p' : hp.uniform('drop1', 0, 0.6),

                #'dense2_num_units' : hp.quniform('dense2', 100, 1000, 20),
                #'dense2_nonlinearity' : hp.uniform('leaky2', 0, 1),
                #'dropout2_p' : hp.uniform('drop2', 0, 0.6),
                
                'output_num_units' : num_classes,
                'output_nonlinearity' : softmax,

                'update' : nesterov_momentum,
              
                'update_learning_rate' : f21(0.03),
                'update_momentum' : f21(0.9),


                'on_epoch_finished' : [
                    AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.99),
        ],
                'verbose' : 1,
                'max_epochs' : 500}


# # Different submission method

# In[14]:

def make_submission_hyper(clf, X_test, ids, encoder, name='../hypersub.csv'):
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


# In[ ]:

# XGBoost polished
subpol = pd.read_csv("../submission.csv").iloc[:, 1:]
logloss_mc(y_valid, subpol)


def objective(hyperparameter):

    hyperparameter['dropoutin_p'] = f21(hyperparameter['dropoutin_p'])
    
    hyperparameter['dense0_nonlinearity'] = LeakyRectify(f21(hyperparameter['dense0_nonlinearity']))
    hyperparameter['dropout0_p'] = f21(hyperparameter['dropout0_p'])

    hyperparameter['dense1_nonlinearity'] = LeakyRectify(f21(hyperparameter['dense1_nonlinearity']))
    hyperparameter['dropout1_p'] = f21(hyperparameter['dropout1_p'])

    #hyperparameter['dense2_nonlinearity'] = LeakyRectify(f21(hyperparameter['dense2_nonlinearity']))
    #hyperparameter['dropout2_p'] = f21(hyperparameter['dropout2_p'])

    mynet = NeuralNet(** hyperparameter)
    mynet.fit(X, y)

    make_submission_hyper(mynet, X_test, ids, encoder)

    sub = pd.read_csv("../hypersub.csv").iloc[:, 1:]
    ll = logloss_mc(y_valid, sub)
    print(ll)
    print(hyperparameter)
    return {'loss' : ll,
            'status' : STATUS_OK}


trials = Trials()

best = fmin(fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=70,
    trials=trials)

print best

# In[ ]:

print trials.trials()

print best