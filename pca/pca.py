'''

File: pca.py
Date: 02.21.15
Author: Hadayat Seddiqi
Description: Straightforward implementation of linear PCA
             on a sample dataset.

'''

import numpy as np
import scipy.linalg as spla
import scipy.io as sio
import matplotlib.pyplot as plt


def pca(data, p, svd=False):
    """
    Do linear PCA and return a reconstruction using the
    @p principle components of an eigendecomposed @data
    matrix.

    Note: This can be done faster for larger datasets
          using scipy.sparse matrices and iterative
          eigenpair-finding methods (e.g. Lanczos).

    Returns: matrix of shape @data.shape.
    """
    # sample means
    means = data.mean(axis=0)
    meanmat = np.tile(means, 
                      (ndata*len(digits),1))
    # covariance matrix
    cov = np.cov(data.T)
    if svd:
        # get the SVD
        evecs, evals, vvecs = spla.svd(cov)
        # keep only some of them
        evecs = evecs[:,:p]
        evals = evals[:p]
    else:
        # get eigendecomposition
        evals, evecs = spla.eigh(cov)
    # sort by eigenvalues
    sortidx = np.argsort(evals)[::-1]
    evals = evals[sortidx]
    evecs = evecs[:,sortidx]
    # keep only some of them
    evals = evals[:p]
    evecs = evecs[:,:p]
    # projection down to p-dimensional PC space
    projection = np.dot(
        evecs.T, 
        (data - meanmat).T
    )
    # reconstruct
    return (np.dot(evecs, projection) + meanmat.T).T

if __name__ == "__main__":
    # set some params
    ndata = 12
    # what do we want to include?
    digits = ['0','1', '7']
    digits = [ str(k) for k in range(0,10,2) ]
    # number of principle components
    p = 20

    # import data
    mnist = sio.loadmat('../datasets/mnist_all.mat')
    # get training data
    data = np.vstack(
        ( mnist['train'+d][:ndata] for d in digits )
    )
    recon = pca(data, p, svd=False)
    # plot everything
    fig, ax = plt.subplots(2*len(digits), ndata)
    fig.suptitle("Actual (top rows) and "+
                  str(p)+"-components projection (bottom rows)")
    plt.gray()
    for row in xrange(0,len(digits)):
        for col in xrange(ndata):
            # original
            ax[2*row,col].imshow(data[row*ndata+col].reshape(28,28))
            # reconstruction
            ax[2*row+1,col].imshow(recon[row*ndata+col].reshape(28,28))
            # get rid of annoying stuff
            ax[2*row,col].set_xticks([])
            ax[2*row+1,col].set_xticks([])
            ax[2*row,col].set_yticks([])
            ax[2*row+1,col].set_yticks([])
    plt.show()
