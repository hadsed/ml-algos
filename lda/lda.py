'''

File: lda.py
Date: 02.23.15
Author: Hadayat Seddiqi
Description: Fisher's linear discriminant analysis for 
             supervised dimensionality reduction.

'''

import numpy as np
import scipy.linalg as spla
import scipy.io as sio
import matplotlib.pyplot as plt


def lda(data, k):
    """
    Do LDA on @data into a lower @k-dimensional
    space and return the corresponding projection matrix.
    @data is a dictionary whose keys contain the classes
    of its values, each of which contains a matrix with
    rows representing data vectors and columns representing
    variables.

    Returns: projection matrix
    """
    # keep a running sum and count for total data mean
    datamean = 0.0
    datacount = 0
    # keep class count, means, and covariance matrices
    classcounts = {}
    classmeans = {}
    classcov = {}
    # loop over data
    for cls, dmat in data.iteritems():
        ccount = dmat.shape[0]
        csum = np.sum(dmat,axis=0)
        # class-specific stuff
        classcounts[cls] = ccount
        classmeans[cls] = csum/float(ccount)
        classcov[cls] = np.cov(dmat.T)
        # all data info
        datamean += csum
        datacount += ccount
    # finish off data mean calculation
    datamean = datamean/datacount
    # compute between-class scatter
    A = np.sum([ classcounts[cls]*np.outer(classmeans[cls]-datamean,
                                           classmeans[cls]-datamean)
                 for cls in classmeans.keys() ],
               axis=0)
    # compute within-class scatter
    B = np.sum([ classcounts[cls]*classcov[cls]
                 for cls in classcov.keys() ],
               axis=0)
    # construct a new basis using (thin) SVD to remove nullspace
    # contributions (it estimates the true rank of the data)
    Q = spla.orth(np.vstack( data[key] for key in data.iterkeys() ).T)
    # rewrite B in this new basis
    print Q.shape, B.shape
    B = np.dot(Q.T, np.dot(B, Q))
    # inverse of cholesky factor
    print spla.cholesky(B)
    Bc = spla.inv(spla.cholesky(B))
    # find the principle eigenvectors
    uvecs, evals, vvecs = spla.svd(np.dot(Bc.T, np.dot(Q.T, A)))
    # return only the first @k vectors for the projection matrix 
    return uvecs[:,:k]

if __name__ == "__main__":
    # set some params
    ndata = 12
    # what do we want to include?
    digits = ['0','1']
    # digits = [ str(k) for k in range(0,10,2) ]
    # number of principle components
    p = 2

    # import data
    mnist = sio.loadmat('../datasets/mnist_all.mat')
    # get training data
    data = dict( (d, mnist['train'+d][:ndata]) 
                 for d in digits )
    # get projection matrix
    proj = lda(data, p)
    # first and second axis projectors
    p0 = proj[:,:0]
    p1 = proj[:,:1]
    # plot everything
    fig, ax = plt.subplots()
    fig.suptitle("LDA for MNIST digits")
    for cls in data.iterkeys():
        for row in data[cls]:
            ax.plot(np.dot(p0.T,row), np.dot(p1.T,row), label="class "+cls)
    plt.legend()
    plt.show()
