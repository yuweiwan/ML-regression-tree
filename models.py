"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class node():
    def __init__(self, d=-1, theta=-1):
        self.d = d
        self.theta = theta
        self.left = None
        self.right = None
        self.label = None


class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """
        # renew each node
        def _fit(X, y, depth):
            # base cases
            if len(y) == 1 or np.std(y) == 0:
                tree_node = node()
                tree_node.label = y[0]
                return tree_node

            if depth == self.max_depth:
                tree_node = node()
                tree_node.label = np.mean(y)
                return tree_node
            # find best theta, feature
            SSE_best = None
            d = 0
            theta = X[0, d]
            for _d in range(self.num_input_features):
                for _theta in np.unique(X[:, _d]):
                    y_l = y[X[:, _d] < _theta]
                    y_r = y[X[:, _d] >= _theta]
                    mu_l = np.mean(y_l)
                    mu_r = np.mean(y_r)
                    SSE = np.sum((y_l - mu_l) ** 2) + np.sum((y_r - mu_r) ** 2)
                    if SSE_best is None:
                        SSE_best = SSE
                        d = _d
                        theta = _theta
                    elif SSE < SSE_best:
                        SSE_best = SSE
                        d = _d
                        theta = _theta

            X_l = X[X[:, d] < theta, :]
            X_r = X[X[:, d] >= theta, :]
            y_l = y[X[:, d] < theta]
            y_r = y[X[:, d] >= theta]

            tree_node = node(d, theta)
            tree_node.left = _fit(X_l, y_l, depth=depth + 1)
            tree_node.right = _fit(X_r, y_r, depth=depth + 1)

            return tree_node

        self.node = _fit(X, y, depth=0)

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """

        def _predict(node, x):
            if node.label is not None:
                return node.label
            elif x[node.d] < node.theta:
                return _predict(node.left, x)
            else:
                return _predict(node.right, x)

        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.node
            prediction[i] = _predict(node, X[i, :])

        return prediction


class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        F0 = RegressionTree(self.num_input_features, 0)  # F0 returns mean(y)
        F0.fit(X=X, y=y)
        self.F = [F0]
        g = np.copy(y)
        for i in range(self.n_estimators):
            if i == 0:
                g -= self.F[-1].predict(X)
            else:
                g -= self.regularization_parameter * self.F[-1].predict(X)
            tree = RegressionTree(self.num_input_features, self.max_depth)
            tree.fit(X=X, y=g)
            self.F.append(tree)

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        predictions = np.zeros((X.shape[0], 1 + self.n_estimators))
        for j in range(self.n_estimators + 1):
            predictions[:, j] = self.F[j].predict(X)
        prediction = predictions[:, 0] + self.regularization_parameter * np.sum(predictions[:, 1:], axis=1)

        return prediction
