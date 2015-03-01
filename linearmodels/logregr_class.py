'''

File: logregr_class.py
Date: 03.01.15
Author: Hadayat Seddiqi
Description: Use logistic regression for MNIST classification.

'''

import numpy as np
import scipy.spatial as spsp
import scipy.io as sio
import matplotlib.pyplot as plt


def logit(x):
    """
    Implement logistic function.
    """
    return 1.0/(1.0 + np.exp(-x))

def softmax(x, w, b):
    """
    Implement softmax:

    p(c=i|x) = e^(w_i.T*x + b_i) / 
               \sum_j^C e^(w_j.T*x + b_j)

    """
    pass

def train(data, labels, test, lr=0.01, steps=100):
    """
    Train a logistic regressor using @data and its
    corresponding @labels, formulating class label
    predictions for @test data. @labels is a vector
    of length @data.shape[1], so that each column in 
    @data corresponds to a training point.
    """
    # initialize weights, one for each dimension
    w = 1e-8*np.random.rand(data.shape[0], 1)
    # bias
    b = 1.0e-8
    # gradient ascent for maximum likelihood
    for k in xrange(steps):
        # dL/dw = \sum_n^N (c^n - \sigma(w^T * x^n + b)) x^n
        gradLw = np.dot((labels-logit(np.dot(w.T, data)+b)), data.T).T
        # dL/db = \sum_n^N (c^n - \sigma(w^T * x^n + b))
        gradLb = np.sum(labels-logit(np.dot(w.T, data)+b))
        # update weights and bias
        w = w + lr*gradLw
        b = b + lr*gradLb
    # now give the predictions for the test data
    return np.asarray(logit(np.dot(w.T, test) + b) > 0.5, dtype=int)

if __name__ == "__main__":
    # number of training points to use for each class
    ntrain = 100
    # number of test points
    ntest = 500
    # learning rate and steps for gradient ascent
    learningrate = 1e-4
    gasteps = 10
    # what do we want to include? (can only choose two)
    digits = ['1','7']
    # import data
    mnist = sio.loadmat('../datasets/mnist_all.mat')
    # get training data
    traindata = np.vstack(
        ( mnist['train'+d][:ntrain] for d in digits )
    ).T
    # labels
    trainlabels = np.array(
        reduce(lambda x,y: x+y,
               [ [int(digits.index(k))]*ntrain for k in digits ] )
    )
    # some test data
    testdata = np.vstack(
        ( mnist['test'+d][:ntest] for d in digits )
    ).T
    # true labels
    testlabels = np.array(
        reduce(lambda x,y: x+y,
               [ [int(digits.index(k))]*ntest for k in digits ] )
    )
    # predicted labels
    pred = train(traindata, trainlabels, testdata, learningrate, gasteps)
    # performance rate as percentage correct
    perf = np.sum(testlabels == pred)/float(ntest*len(digits))
    print("Success rate (percentage):", perf)
