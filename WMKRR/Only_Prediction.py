# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

class CombinedKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, gamma1=0.1, gamma2=0.1, weight=0.5):
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weight = weight
        self.krr = KernelRidge(alpha=alpha, kernel="precomputed")

    # Gaussian kernel
    def _gaussian_kernel(self, X1, X2, gamma):
        return np.exp(-gamma * ((X1[:, np.newaxis] - X2[np.newaxis, :]) ** 2).sum(axis=2))

    # Training model
    def fit(self, X, y, n_features1, n_features2):
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        X1, X2 = X[:, :self.n_features1], X[:, self.n_features1:]
        K1 = self._gaussian_kernel(X1, X1, self.gamma1)
        K2 = self._gaussian_kernel(X2, X2, self.gamma2)
        K = self.weight * K1 + (1 - self.weight) * K2
        self.krr.fit(K, y)
        self.X_fit_ = X
        return self   

    # Prediction
    def predict(self, X):
        X1, X2 = X[:, :self.n_features1], X[:, self.n_features1:]
        K1 = self._gaussian_kernel(X1, self.X_fit_[:, :self.n_features1], self.gamma1)
        K2 = self._gaussian_kernel(X2, self.X_fit_[:, self.n_features1:], self.gamma2)
        K = self.weight * K1 + (1 - self.weight) * K2
        return self.krr.predict(K)


# Load data
X1_train = np.loadtxt('X1.txt')
X2_train = np.loadtxt('X2.txt')
y_train = np.loadtxt('y.txt')
X_train = np.hstack([X1_train, X2_train])

n_features1 = X1_train.shape[1]
n_features2 = X2_train.shape[1]

X1_test = np.loadtxt('X1_test.txt')
X2_test = np.loadtxt('X2_test.txt')
X_test = np.hstack([X1_test, X2_test])

# Manually setting the optimal parameters
best_params = {'alpha': 1.0, 'gamma1': 0.001, 'gamma2': 0.0001, 'weight': 0.5}     #Note: The parameter values need to be modified here based on the tuning results.

# Prediction using optimal parameters
best_model = CombinedKernelRidge(**best_params)
best_model.fit(X_train, y_train, n_features1=n_features1, n_features2=n_features2)
y_pred = best_model.predict(X_test)

# Save prediction results
np.savetxt('y_test_pred.txt', y_pred)

