import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""
        p_xy = np.ones((d,k))
        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        for j in range(d):
            for i in range(k):
                numerator = X[y==i,:]
                p_xy[j,i] = np.mean(numerator[:, j])
                print(p_xy[j,i])



        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        n, d = X.shape  # Number of examples (n) and features (d)
        k = self.num_classes  # Number of class labels

        # Compute prior probabilities p(y)
        counts = np.bincount(y, minlength=k)  # Ensure all classes are counted
        p_y = counts / n

        # Compute conditional probabilities p(x_j = 1 | y = c) with Laplace smoothing
        p_xy = np.ones((d, k))

        for c in range(k):
            X_c = X[y == c]  # Extract examples where label is c
            count_y_c = len(X_c)  # Total occurrences of class c

            for j in range(d):
                count_xj_1 = np.sum(X_c[:, j])  # Count of x_j = 1 given y = c
                p_xy[j, c] = (count_xj_1 + self.beta) / (count_y_c + 2 * self.beta)
                print(p_xy[j,c])
        self.p_y = p_y
        self.p_xy = p_xy

