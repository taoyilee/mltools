import numpy as np
import mltools as ml
from .base import classifier
from .utils import toIndex
from numpy import asarray as arr
from numpy import atleast_2d as twod
import scipy.special


# import line_profiler
# import atexit
#
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class linearClassify(classifier):
    """A simple linear classifier

    Attributes:
        classes : a list of the possible class labels
        theta   : linear parameters of the classifier
                  (1xN or CxN numpy array, where N=# features, C=# classes)

    Note: currently specialized to logistic loss
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for linearClassify object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        self.theta = np.array([])

        if len(args) or len(kwargs):  # if we were given optional arguments,
            self.train(*args, **kwargs)  # just pass them through to "train"

    def __repr__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
            len(self.theta), self.theta)
        return str_rep

    def __str__(self):
        str_rep = 'linearClassify model, {} features\n{}'.format(
            len(self.theta), self.theta)
        return str_rep

    def predictSoft(self, X):
        """
        This method makes a "soft" linear classification predition on the data
        Uses a (multi)-logistic function to convert linear response to [0,1] confidence

        Parameters
        ----------
        X : M x N numpy array
            M = number of testing instances; N = number of features.
        """
        resp = X.dot(self.theta[1:].T) + self.theta[0]  # linear response (MxC)
        prob = np.exp(resp)
        if resp.ndim == 1 or resp.shape[1] == 1:  # binary classification (C=1)
            prob /= prob + 1.0  # logistic transform (binary classification; C=1)
            prob = np.stack((1 - prob, prob), axis=1)  # make a column for each class
        else:
            prob /= np.sum(prob, axis=1)  # normalize each row (for multi-class)

        return prob

    """
    Define "predict" here if desired (or just use predictSoft + argmax by default)
    """

    # @profile
    def train(self, X, Y, reg=0.0, initStep=1.0, stopTol=1e-4, stopIter=5000):
        """
        Train the linear classifier.
        """
        m, n = X.shape
        if Y.shape[0] != m:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        self.theta = np.random.randn(n + 1)
        Y01 = toIndex(Y, self.classes)  # convert Y to "index" (binary: 0 vs 1)
        it = 0
        last_jsur = 1e6
        while True:
            step = (2.0 * initStep) / (2.0 + it)  # common 1/iter step size change
            for i in range(m):  # for each data point
                sigx = scipy.special.expit(X[i, :].dot(self.theta[1:].T) + self.theta[0])
                raw_gradient = (sigx - Y01[i]) * X[i, :]
                gradi = raw_gradient + reg * self.theta[1:]
                self.theta[1:] -= step * gradi
                self.theta[0] -= step * reg * self.theta[0]

            jsur = self.nll(X, Y) + reg * np.sum(self.theta ** 2)
            it += 1
            if it > stopIter or abs(jsur - last_jsur) < stopTol:
                return
            last_jsur = jsur

    def lossLogisticNLL(self, X, Y, reg=0.0):
        M, N = X.shape
        P = self.predictSoft(X)
        J = - np.sum(np.log(P[range(M), Y[:]]))  # assumes Y=0...C-1
        Y = ml.to1ofK(Y, self.classes)
        DJ = NotImplemented  ##- np.sum( P**Y
        return J, DJ
