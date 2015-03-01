'''

File: kernel_1d.py
Date: 02.28.15
Author: Hadayat Seddiqi
Description: Use the kernel formulation for
             least squares regression.

'''

import numpy as np
import scipy.spatial as spsp
import matplotlib.pyplot as plt


def rbf(x, y, theta):
    """
    Radial-basis kernel function. Returns the kernel
    matrix for a set of points @x and @y (they can
    be the same vector though).
    """
    sf2, l2 = theta
    # scipy wants 2D arrays
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    K = spsp.distance.cdist(x, y, 'sqeuclidean')
    return sf2*np.exp(-K/(2.0*l2))

def per(x, y, theta):
    """
    Periodic kernel function. Returns the kernel
    matrix for a set of points @x and @y (they can
    be the same vector though).
    """
    sf2, l2, p = theta
    # scipy wants 2D arrays
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))
    K = spsp.distance.cdist(x, y, 'euclidean')
    return sf2*np.exp(-2.0*np.sin(np.pi*K/p)/l2)

def kernel_leastsq_train(xtrain, ytrain, kernel, theta, r=0.0):
    """
    Train a least-squares regression model using the
    given training data (@xtrain, @ytrain) and a kernel
    function @kernel with hyperparameters given by @theta.
    Return a set of weights of length equal to @ytrain.shape. 
    @r is a regularization parameter.
    """
    # kernel matrix for training data
    K = kernel(xtrain, xtrain, theta)
    # new test points
    ks = kernel(xtest, xtrain, theta)
    # regularization
    if r > 0.0:
        K += r*np.identity(xtrain.size)
    # ys = ks^T * (K + r*I)^-1 * y
    return np.inner(ks,np.dot(np.linalg.inv(K),ytrain))

if __name__=="__main__":
    # sine-function data
    xtrain = np.linspace(0,1,100)
    ytrain = np.sin(10.0*xtrain) + \
             np.random.normal(0.0,3e-1,size=xtrain.size)
    # xval = np.linspace(0,1,10)
    # yval = np.sin(10.0*xval) + \
    #        np.random.normal(0.0,1e-1,size=xval.size)
    xtest = np.linspace(0.0,1.0,200)
    # hyperparameters
    theta_rbf = [1.0e-1, 5e-2]
    theta_per = [1.0e-2, 1.0e-1, 5.0e1/np.pi]
    # regularization is important
    r = 0.01
    ytest_rbf = kernel_leastsq_train(xtrain, ytrain, rbf, theta_rbf, r)
    ytest_per = kernel_leastsq_train(xtrain, ytrain, per, theta_per, r)
    # try no regularizing term
    ytest_rbf_nor = kernel_leastsq_train(xtrain, ytrain, rbf, theta_rbf)
    ytest_per_nor = kernel_leastsq_train(xtrain, ytrain, per, theta_per)
    # plot
    plt.plot(xtrain, ytrain, 'kx', label="data", markersize=7)
    plt.plot(xtest, ytest_rbf, '-r', linewidth=1, label="rbf")
    plt.plot(xtest, ytest_per, '-g', linewidth=1, label="periodic")
    plt.plot(xtest, ytest_rbf_nor, '-.y', linewidth=1, label="rbf no-reg")
    plt.plot(xtest, ytest_per_nor, '--b', linewidth=1, label="periodic no-reg")
    plt.plot(xtest, np.sin(10.0*xtest), '-k', linewidth=2, label="true")
    plt.ylim([-1.5,1.5])
    plt.legend(loc=4)
    plt.show()
