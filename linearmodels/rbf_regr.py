'''

File: rbf_1d.py
Date: 02.28.15
Author: Hadayat Seddiqi
Description: Use an RBF function this time.

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
    # sine-function data
    xtrain = np.linspace(0,1,30)
    ytrain = np.sin(10.0*xtrain) + \
             np.random.normal(0.0,1e-1,size=xtrain.size)
    xval = np.linspace(0,1,10)
    yval = np.sin(10.0*xval) + \
           np.random.normal(0.0,1e-1,size=xval.size)
    xtest = np.linspace(0.05,0.95,30)
    # cubic function
    fcub = lambda x: np.array([1.0, x, x**2, x**3])
    # quartic
    fquart = lambda x: np.array([1.0, x, x**2, x**3, x**4])
    # find the optimal parameters for the rbf function
    mu = np.linspace(0,1,5)
    alphas = []
    err = []
    # loop over possible alpha values
    for alpha in np.concatenate((np.linspace(0.001,0.1,50),
                                 np.linspace(0.1,1.0,50))):
        # construct the function
        fcand = lambda x: np.array(
            [ np.exp(-1.0/(2*alpha**2)*
                     np.inner(x-mu[k],x-mu[k]))
              for k in xrange(mu.size) ]
        )
        # and its predictions on validation points
        ytry = np.array([
            np.dot(leastsq_train(xtrain, ytrain, fcand),
                   fcand(xval[k]))
            for k in xrange(xval.size)
        ])
        # calculate the error and keep it
        err.append(((ytry - yval)**2).mean(axis=0))
        alphas.append(alpha)
    # plot validation error for alpha parameter
    plt.plot(alphas, err, 'b')
    plt.xlabel("a")
    plt.ylabel("MSE")
    plt.show()
    # rbf function (optimized)
    alpha = alphas[np.argmin(err)]
    frbf = lambda x: np.array([ np.exp(-1.0/(2*alpha**2)*
                                       np.inner(x-mu[k],x-mu[k]))
                                for k in xrange(mu.size) ])
    # get the models
    wcub = leastsq_train(xtrain, ytrain, fcub)
    wquart = leastsq_train(xtrain, ytrain, fquart)
    wrbf = leastsq_train(xtrain, ytrain, frbf)
    # test outputs
    ytest_cub = np.array([ np.dot(wcub, fcub(xtest[k])) 
                           for k in xrange(xtest.size) ])
    ytest_quart = np.array([ np.dot(wquart, fquart(xtest[k])) 
                             for k in xrange(xtest.size) ])
    ytest_rbf = np.array([ np.dot(wrbf, frbf(xtest[k])) 
                           for k in xrange(xtest.size) ])
    # plot
    plt.plot(xtrain, ytrain, 'kx', label="data", markersize=7)
    plt.plot(xtest, ytest_cub, '-k', linewidth=1, label="cubic fit")
    plt.plot(xtest, ytest_quart, '-g', linewidth=1, label="quartic fit")
    plt.plot(xtest, ytest_rbf, '-r', linewidth=1, label="rbf kernel")
    plt.plot(xtest, np.sin(10.0*xtest), '-b', linewidth=1, label="true")
    plt.legend(loc=4)
    plt.show()
