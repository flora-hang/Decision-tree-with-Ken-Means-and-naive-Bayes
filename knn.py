"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n,d = X_hat.shape
        """YOUR CODE HERE FOR Q1"""
        neighbours = utils.euclidean_dist_squared(X_hat, self.X)

        y_p = np.zeros(n)
        for i in range(n):
            temp = np.argsort(neighbours[i])
            y_p[i] = utils.mode(self.y[temp[:self.k]])

        return y_p
