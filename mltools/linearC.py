import numpy as np
import mltools as ml
from .base import classifier
from .utils import toIndex
from numpy import asarray as arr
from numpy import atleast_2d as twod
import scipy.special
import matplotlib.pyplot as plt


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

    def __init__(self, theta=None, *args, **kwargs):
        """
        Constructor for linearClassify object.

        Parameters: Same as "train" function; calls "train" if available

        Properties:
           classes : list of identifiers for each class
           theta   : linear coefficients of the classifier; numpy array
                      shape (1,N) for binary classification or (C,N) for C classes
        """
        self.classes = []
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.array([])

        if len(args) or len(kwargs):  # if we were given optional arguments,
            self.train(*args, **kwargs)  # just pass them through to "train"

    def plotBoundary(self, X, Y):
        if len(self.theta) != 3: raise ValueError('Data & model must be 2D')
        ax = X.min(0), X.max(0)
        ax = (ax[0][0], ax[1][0], ax[0][1], ax[1][1])
        x1b = np.array([ax[0], ax[1]])
        x2b = (-self.theta[1] * x1b - self.theta[0]) / self.theta[2]
        A = Y == self.classes[0]
        plt.plot(X[A, 0], X[A, 1], 'b.', label="Data Y=+1")
        plt.plot(X[~A, 0], X[~A, 1], 'r.', label="Data Y=-1")
        plt.plot(x1b, x2b, 'k-', label="Classifier")
        plt.axis(ax)
        plt.legend()
        plt.draw()

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
        logit = scipy.special.expit(resp)
        if resp.ndim == 1 or resp.shape[1] == 1:  # binary classification (C=1)
            logit = np.stack((1 - logit, logit), axis=1)  # make a column for each class
        else:
            logit /= np.sum(logit, axis=1)  # normalize each row (for multi-class)

        return logit

    """
    Define "predict" here if desired (or just use predictSoft + argmax by default)
    """

    # @profile
    def train(self, X, Y, reg=0.0, initStep=1.0, stopTol=1e-4, stopIter=5000, rate_decay=0.5, patience=3, minlr=1e-7):
        """
        Train the linear classifier.
        """
        m, n = X.shape
        XX = np.hstack((np.ones((m, 1)), X))
        if Y.shape[0] != m:
            raise ValueError("Y must have the same number of data (rows) as X")
        self.classes = np.unique(Y)
        if len(self.classes) != 2:
            raise ValueError("Y should have exactly two classes (binary problem expected)")
        if len(self.theta) != n + 1:
            self.theta = np.random.randn(n + 1)
        Y01 = toIndex(Y, self.classes)  # convert Y to "index" (binary: 0 vs 1)
        it = 0
        jsur = []
        last_jsur = 1e6
        lr = []
        step = initStep
        wait_cntr = patience
        while True:
            lr.append(step)
            for i in range(m):  # for each data point
                sigx = scipy.special.expit(XX[i, :].dot(self.theta.T))
                raw_gradient = (sigx - Y01[i]) * XX[i, :]
                self.theta -= step * (raw_gradient + reg * self.theta)

            jsur.append(self.nll(X, Y) + reg * np.sum(self.theta ** 2))
            it += 1
            delta_jsur = abs(jsur[-1] - last_jsur)
            # print(f"jsur/delta = {last_jsur:.3e}/{delta_jsur:.3e} {wait_cntr}")
            if wait_cntr == 0 and delta_jsur < last_jsur * 0.01:
                step = max(minlr, step * rate_decay)
                wait_cntr = patience
                # print(f"Reducing stepsize to {step}")
            if it > stopIter or (it > 15 and delta_jsur < stopTol):
                return jsur, lr, it
            if wait_cntr > 0:
                wait_cntr -= 1
            last_jsur = jsur[-1]
