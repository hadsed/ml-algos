'''

File: ls_1d.py
Date: 02.28.15
Author: Hadayat Seddiqi
Description: Simple least-squares fitting on 1D data.

'''

import numpy as np
import matplotlib.pyplot as plt


def leastsq_train(xtrain, ytrain, phi, r=0.0):
    """
    Train a least-squares regression model using the
    given training data (@xtrain, @ytrain) and a mapping
    function @phi. Return a set of weights of length
    equal to @ytrain.shape.
    """
    # sum of outer products
    phi_outer = np.sum([ np.outer(phi(xtrain[k]), phi(xtrain[k]))
                         for k in xrange(xtrain.size) ],
                       axis=0)
    # what we should get when w.T*phi(x)
    yp = np.sum([ ytrain[k]*phi(xtrain[k])
                  for k in xrange(xtrain.size) ],
                axis=0)
    # regularization
    if r > 0.0:
        phi_outer += r*np.identity(phi_outer.shape[0])
    # return w = phi_outer^-1 * yp
    return np.linalg.solve(phi_outer, yp)

if __name__=="__main__":
    # setup the data
    xtrain = np.array([88.6, 71.6, 93.3, 84.3, 80.6, 
                       75.2, 69.7, 82.0, 69.4, 83.3, 
                       79.6, 82.6, 80.6, 83.5, 76.3])
    ytrain = np.array([20.0, 16.0, 19.8, 18.4, 17.1, 
                       15.5, 14.7, 17.1, 15.4, 16.2, 
                       15.0, 17.2, 16.9, 17.0, 14.4])
    # test grid
    xtest = np.linspace(65, 100, 100)
    # linear function
    flin = lambda x: np.array([1.0, x])
    # cubic function
    fcub = lambda x: np.array([1.0, x, x**2, x**3])
    # get the models
    wlin = leastsq_train(xtrain, ytrain, flin)
    wcub = leastsq_train(xtrain, ytrain, fcub)
    wcub_reg = leastsq_train(xtrain, ytrain, fcub, 1.0)
    # test outputs
    ytest_lin = np.array([ np.dot(wlin, flin(xtest[k])) 
                           for k in xrange(xtest.size) ])
    ytest_cub = np.array([ np.dot(wcub, fcub(xtest[k])) 
                           for k in xrange(xtest.size) ])
    ytest_cub_reg = np.array([ np.dot(wcub_reg, fcub(xtest[k])) 
                               for k in xrange(xtest.size) ])
    # plot
    plt.plot(xtrain, ytrain, 'kx', label="data", markersize=7)
    plt.plot(xtest, ytest_lin, '-r', linewidth=1, label="linear fit")
    plt.plot(xtest, ytest_cub, '-b', linewidth=1, label="cubic fit")
    plt.plot(xtest, ytest_cub_reg, '-g', 
             linewidth=1, label="cubic fit (regularized)")
    plt.xlabel("Temperature (F)")
    plt.ylabel("Chirps Per Second")
    plt.legend(loc=4)
    plt.show()
