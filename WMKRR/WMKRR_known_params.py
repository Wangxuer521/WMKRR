# -*- coding: utf-8 -*-
import gc
import argparse
import sys
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge

class CombinedKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        alpha=1.0, 
        gamma1=0.0001, 
        gamma2=0.001, 
        weight=0.4
    ):
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weight = weight
        self.krr = KernelRidge(alpha=alpha, kernel="precomputed")

    # Gaussian kernel
    def _gaussian_kernel(self, X1, X2, gamma):
        pairwise_sq_dists = (
            -2 * np.dot(X1, X2.T)
            + np.sum(X1**2, axis=1)[:, np.newaxis]
            + np.sum(X2**2, axis=1)
        )
        
        return np.exp(-gamma * pairwise_sq_dists)

    # Training model
    def fit(
        self, 
        X, 
        y, 
        n_features1, 
        n_features2
    ):
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        
        X1 = X[:, :self.n_features1]
        X2 = X[:, self.n_features1:]
        
        K1 = self._gaussian_kernel(X1, X1, self.gamma1)
        K2 = self._gaussian_kernel(X2, X2, self.gamma2)
        K = self.weight * K1 + (1 - self.weight) * K2
        
        self.krr.fit(K, y)
        self.X_fit_ = X
        
        return self   

    # Prediction
    def predict(self, X):
        X1 = X[:, :self.n_features1]
        X2 = X[:, self.n_features1:]
        
        K1 = self._gaussian_kernel(
            X1, 
            self.X_fit_[:, :self.n_features1], 
            self.gamma1
        )
        
        K2 = self._gaussian_kernel(
            X2, 
            self.X_fit_[:, self.n_features1:], 
            self.gamma2
        )
        
        K = self.weight * K1 + (1 - self.weight) * K2
        
        return self.krr.predict(K)

# Parse CLI args
def parse_args():
    parser = argparse.ArgumentParser(description="WMKRR prediction with known parameters")
    
    parser.add_argument(
        "--alpha",  
        type=float, 
        required=True, 
        help="Regularization factor (>0), e.g., 0.5"
    )
    
    parser.add_argument(
        "--gamma1", 
        type=float, 
        required=True, 
        help="Gamma parameter of the first RBF kernel (K1) (>0), e.g., 5e-05"
    )
    
    parser.add_argument(
        "--gamma2", 
        type=float, 
        required=True, 
        help="Gamma parameter of the second RBF kernel (K2) (>0), e.g., 9e-04"
    )
    
    parser.add_argument(
        "--weight", 
        type=float, 
        required=True, 
        help="Weight of the first RBF kernel (K1) ([0,1]), e.g., 0.9"
    )
    
    args = parser.parse_args()

    # simple validation
    if args.alpha <= 0 or args.gamma1 <= 0 or args.gamma2 <= 0:
        parser.error("--alpha, --gamma1, --gamma2 must all be > 0")
        
    if not (0.0 <= args.weight <= 1.0):
        parser.error("--weight must be within [0, 1]")
        
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load data
    X1_train = np.loadtxt('./example_data/X1.txt')
    X2_train = np.loadtxt('./example_data/X2.txt')

    X_train = np.hstack([X1_train, X2_train])
    y_train = np.loadtxt('./example_data/y.txt')

    n_features1 = X1_train.shape[1]
    n_features2 = X2_train.shape[1]

    X1_test = np.loadtxt('./example_data/X1_test.txt')
    X2_test = np.loadtxt('./example_data/X2_test.txt')
    X_test = np.hstack([X1_test, X2_test])

    # Release unnecessary data
    del X1_train, X2_train, X1_test, X2_test
    gc.collect()

    # Parameters from CLI
    best_params = {
        'alpha':  args.alpha,
        'gamma1': args.gamma1,
        'gamma2': args.gamma2,
        'weight': args.weight
    }
    
    print("Using parameters:", best_params, flush=True)

    # Prediction using provided parameters
    best_model = CombinedKernelRidge(**best_params)
    
    best_model.fit(
        X_train, 
        y_train, 
        n_features1=n_features1, 
        n_features2=n_features2
    )
    
    y_pred = best_model.predict(X_test)

    # Save prediction results
    np.savetxt('y_test_pred.txt', y_pred)

