'''

File: knn.py
Date: 02.21.15
Author: Hadayat Seddiqi
Description: Implementation of k-nearest neighbor classification.

'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter


def sqeuclidean(x,y):
    """
    The squared Euclidean distance function between vectors
    @x and @y.
    """
    if x.shape != y.shape:
        print("euclidean: X and Y must be the same shape.")
    return np.inner(x-y,x-y)

def mahalanobis(x,y,s):
    """
    The Mahalanobis distance function between @x and @y:

    d(x,y) = (x-y)^T Z^-1 (x-y)

    where Z^-1 is the covariance matrix.
    """
    if x.shape != y.shape:
        print("euclidean: X and Y must be the same shape.")
    return np.inner(x-y, np.dot(s, x-y))

def cov(data):
    """
    Output the covariance matrix of @data, where data vectors
    are rows, columns are its different components.
    """
    means = data.sum(axis=0)/float(data.shape[0])
    # subtract off the means
    data = data - np.tile(means, (data.shape[0],1))
    # covmat = np.sum([ np.outer(d,d) for d in data ], 
    #                 axis=0) / (data.shape[0]-1)
    covmat = np.dot(data.T, data) / (data.shape[0]-1)
    return covmat

def classify_knn(test, data, labels, k, dist, plot=False):
    """
    Classify datapoints given in @test using @k-nearest neighbors
    from a training dataset @data. @labels should be a vector of
    length @data.shape[0] for the class labels. @dist should be 
    either "sqeuclidean" or "mahalanobis". @plot shows the test 
    vector and prints the predicted and actual class in the title.

    Returns: vector of classes with same shape as @test.shape[0]
    """
    # keep the predicted class labels somewhere
    classes = np.empty(test.shape[0])
    # precalculate covariance matrix if needed
    if dist == 'mahalanobis':
        covmat = cov(data)
    # get test vectors
    for itvec, tvec in enumerate(test):
        if dist == 'sqeuclidean':
            if k == 1:
                classes[itvec] = labels[np.argmin([ sqeuclidean(tvec, t) for t in data ])]
            else:
                # get the top @k label indices
                candsidx = np.argsort([ sqeuclidean(tvec, t) for t in data ])[0:k]
                # this gives us the actual labels
                klabels = list(labels[candsidx])
                # unique labels to count for
                possibles = list(set(klabels))
                # get max count label
                classes[itvec] = possibles[np.argmax([ klabels.count(c) 
                                                       for c in possibles ])]
        elif dist == 'mahalanobis':
            if k == 1:
                classes[itvec] = labels[np.argmin([ mahalanobis(tvec, t, covmat) for t in data ])]
            else:
                # get the top @k label indices
                candsidx = np.argsort([ mahalanobis(tvec, t, covmat) for t in data ])[0:k]
                # this gives us the actual labels
                klabels = list(labels[candsidx])
                # unique labels to count for
                possibles = list(set(klabels))
                # get max count label
                classes[itvec] = possibles[np.argmax([ klabels.count(c) 
                                                       for c in possibles ])]
        if plot:
            plt.imshow(tvec.reshape(28,28))
            plt.gray()
            plt.show()
    return classes


if __name__ == "__main__":
    # set some params
    ntrain = 100
    ntest = 50
    kneighbors = 10
    # mahal. gets 0.93 success, sqeuc. only 0.33
    digits = ['0','1', '7']
    # but then here mahal. gets 0.66 success, sqeuc. only 0.1
    # digits = [ str(k) for k in xrange(10) ]

    # import data
    mnist = sio.loadmat('../datasets/mnist_all.mat')
    # get training data
    traindata = np.vstack(
        ( mnist['train'+d][:ntrain] for d in digits )
    )
    # labels
    trainlabels = np.array(
        reduce(lambda x,y: x+y,
               [ [int(k)]*ntrain for k in digits ] 
        )
    )
    # some test data
    testdata = np.vstack(
        ( mnist['test'+d][:ntest] for d in digits )
    )
    # true labels
    testlabels = np.array(
        reduce(lambda x,y: x+y,
               [ [int(k)]*ntest for k in digits ] 
        )
    )
    
    # gather some performance data as function of # neighbors
    sqeu_perf = np.empty(kneighbors)
    mahal_perf = np.empty(kneighbors)
    for knbs in xrange(1,kneighbors):
        # try to predict the labels
        sqeu_pred = classify_knn(test=testdata, 
                                 data=traindata, 
                                 labels=trainlabels,
                                 k=knbs,
                                 dist='sqeuclidean')
        sqeu_perf[knbs] = np.sum(testlabels == sqeu_pred)/float(ntest*len(digits))
        # now with mahalanobis
        mahal_pred = classify_knn(test=testdata, 
                                  data=traindata, 
                                  labels=trainlabels,
                                  k=knbs,
                                  dist='mahalanobis')
        mahal_perf[knbs] = np.sum(testlabels == mahal_pred)/float(ntest*len(digits))
    # plot our results
    plt.plot(range(kneighbors), sqeu_perf, label="sq. euclidean")
    plt.plot(range(kneighbors), mahal_perf, label="mahalanobis")
    plt.title("Success Rate vs. K with KNN on some MNIST digits")
    plt.ylabel("Success Rate")
    plt.xlabel("Number of Neighbors")
    plt.legend()
    plt.show()
