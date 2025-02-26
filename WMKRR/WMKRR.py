import time
import gc, os
import logging
import numpy as np
from skopt import BayesSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from numbers import Real as NumbersReal
from skopt.space import Real as SkoptReal
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_array, check_X_y
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pearson_corr_coef(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    diff_true = y_true - mean_true
    diff_pred = y_pred - mean_pred
    diff_product = diff_true * diff_pred
    numerator = np.sum(diff_product)
    diff_true2 = diff_true ** 2
    diff_pred2 = diff_pred ** 2
    sum_diff_true2 = np.sum(diff_true2)
    sum_diff_pred2 = np.sum(diff_pred2)
    pairwise_product = sum_diff_true2 * sum_diff_pred2
    denominator = np.sqrt(pairwise_product)
    
    if denominator == 0:
        logging.warning("Pearson correlation denominator is zero. Returning 0.")
        return 0
    else:
        result = numerator / denominator
        return result     

class PearsonCorrScorer:
    def __init__(self, greater_is_better=True):
        self.greater_is_better = greater_is_better
    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        score = pearson_corr_coef(y, y_pred)
        if self.greater_is_better:
            return score
        else:
            return -score

def load_data(file_name):
    logging.info(f"Loading data from {file_name}")
    try:
        data = np.loadtxt(file_name)
        logging.info(f"Data loaded successfully from {file_name}")
        
    except Exception as e:
        logging.error(f"Error loading data from {file_name}: {e}")
        raise

    return data

def clean_memory(*args):
    for arg in args:
        logging.info(f"Cleaning memory...")
        del arg
    gc.collect()
    logging.info("Memory cleaned, garbage collection completed.")

def write_params_to_file(file_name, params):
    logging.info(f"Writing parameters to {file_name}")
    try:
        with open(file_name, 'w') as f:
            for k, v in params.items():
                k_str = str(k)
                v_str = str(v)
                f.write(k_str + '\t' + v_str + '\n')
        logging.info(f"Parameters written to {file_name} successfully.")
        
    except Exception as e:
        logging.error(f"Error writing parameters to {file_name}: {e}")
        raise       

def save_predictions_to_file(file_name, ids, predictions):
    logging.info(f"Saving predictions to {file_name}")
    try:
        with open(file_name, 'w') as f:
            pair = zip(ids, predictions)
            for item in pair:
                id = item[0]
                pred = item[1]
                id_str = str(id)
                pred_str = str(pred)
                line = id_str + '\t' + pred_str + '\n'
                f.write(line)
        logging.info(f"Predictions saved to {file_name} successfully.")
        
    except Exception as e:
        logging.error(f"Error saving predictions to {file_name}: {e}")
        raise

class KernelRidge(MultiOutputMixin, RegressorMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "alpha": [
            Interval(NumbersReal, 0, None, closed="left"), 
            "array-like"
        ],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS.keys()) | {"precomputed"}),
            callable,
        ],
        "gamma": [
            Interval(NumbersReal, 0, None, closed="left"), 
            None
        ],
        "degree": [
            Interval(NumbersReal, 0, None, closed="left")
        ],
        "coef0": [
            Interval(NumbersReal, None, None, closed="neither")
        ],
        "kernel_params": [dict, None],
    }
    
    def __init__(
        self,
        alpha=1,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "gamma": self.gamma, 
                "degree": self.degree, 
                "coef0": self.coef0
            }
        kernel_result = pairwise_kernels(
            X, Y, 
            metric=self.kernel, 
            filter_params=True,
            **params
        )
        
        return kernel_result
        
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if self.kernel == "precomputed":
            tags.input_tags.pairwise = True
        else:
            tags.input_tags.pairwise = False
        return tags
    @_fit_context(prefer_skip_nested_validation=True)
    
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(
            X, y, 
            accept_sparse=("csr", "csc"), 
            multi_output=True, 
            y_numeric=True
        )
        
        if sample_weight is not None:
            if not isinstance(sample_weight, float):
                sample_weight = _check_sample_weight(sample_weight, X)
        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)
        ravel = False
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True
        copy = self.kernel == "precomputed"
        self.dual_coef_ = _solve_cholesky_kernel(
            K, y, 
            alpha, 
            sample_weight, 
            copy
        )
        
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()
        self.X_fit_ = X
        
        return self
        
    def predict(self, X):
        X = check_array(X, accept_sparse=("csr", "csc"))
        K = self._get_kernel(X, self.X_fit_)
        prediction = np.dot(K, self.dual_coef_)

        return prediction
        
class CombinedKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(
        self, 
        alpha=1.0, 
        gamma1=0.0001, 
        gamma2=0.0001, 
        weight=0.5
    ):
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weight = weight
        self.krr = KernelRidge(
            alpha=alpha, 
            kernel="precomputed",
            gamma=None,
            degree=3,
            coef0=1,
            kernel_params=None
        ) 
        
    def _gaussian_kernel(self, X1, X2, gamma):
        logging.info("Calculating Gaussian kernel between X1 and X2")
        X2_T = X2.T
        dot_product = np.dot(X1, X2_T)
        dot_product_scaled = -2 * dot_product
        logging.debug(f"Dot product shape: {dot_product.shape}")
        X1_squared = X1**2
        sum_X1_squared = np.sum(X1_squared, axis=1)
        new_axis = np.newaxis
        sum_X1_squared_col = sum_X1_squared[:, new_axis]
        X2_squared = X2**2
        sum_X2_squared = np.sum(X2_squared, axis=1)
        pairwise_sq_dists = dot_product_scaled + sum_X1_squared_col + \
                            sum_X2_squared
        scaled_dists = gamma * pairwise_sq_dists
        kernel = np.exp(-scaled_dists)
        logging.debug(f"Kernel matrix construction completed.")
        
        return kernel
        
    def fit(self, X, y, n_features1, n_features2):
        start_time = time.time()
        logging.info("Fitting the CombinedKernelRidge model")
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        n1 = self.n_features1
        X1 = X[:, :n1]
        X2 = X[:, n1:]
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        
        K1 = self._gaussian_kernel(X1, X1, gamma1)
        K2 = self._gaussian_kernel(X2, X2, gamma2)
        w1 = self.weight
        w2 = 1 - self.weight
        weighted_K1 = w1 * K1
        weighted_K2 = w2 * K2
        K = weighted_K1 + weighted_K2
        
        self.krr.fit(K, y)
        self.X_fit_ = X
        end_time = time.time()
        fit_time = end_time - start_time
        logging.info(f"Model fitting completed in {fit_time} seconds.")
        
        return self
        
    def predict(self, X):
        start_time = time.time()
        logging.info("Making predictions using the CombinedKernelRidge model")
        n1 = self.n_features1
        X1 = X[:, :n1]
        X2 = X[:, n1:]
        X_fit_1 = self.X_fit_[:, :n1]
        X_fit_2 = self.X_fit_[:, n1:]
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        
        K1 = self._gaussian_kernel(X1, X_fit_1, gamma1)
        K2 = self._gaussian_kernel(X2, X_fit_2, gamma2)
        w1 = self.weight
        w2 = 1 - self.weight
        weighted_K1 = w1 * K1
        weighted_K2 = w2 * K2
        K = weighted_K1 + weighted_K2
        
        prediction = self.krr.predict(K)
        end_time = time.time()
        pred_time = end_time - start_time
        logging.info(f"Predictions completed in {pred_time} seconds.")
        
        return prediction
        
# Pearson correlation coefficient score         
pearson_scorer = PearsonCorrScorer(greater_is_better=True)

# load in training set data
X1_train = load_data('./example_data/X1.txt')
X2_train = load_data('./example_data/X2.txt')
y_train = load_data('./example_data/y.txt')
X_train = np.hstack([X1_train, X2_train])

# number of features
n_features1 = X1_train.shape[1]
n_features2 = X2_train.shape[1]

# load in test set data
X1_test = load_data('./example_data/X1_test.txt')
X2_test = load_data('./example_data/X2_test.txt')
X_test = np.hstack([X1_test, X2_test])

# release memory
clean_memory(X1_train,X2_train,X1_test,X2_test)

# define parameter space
alpha_param = SkoptReal(0.01, 1, prior="log-uniform")
gamma1_param = SkoptReal(1e-8, 1e-3, prior="log-uniform")
gamma2_param = SkoptReal(1e-8, 1e-3, prior="log-uniform")
weight_param = SkoptReal(0.1, 0.9)

param_space = {
    "alpha": alpha_param,
    "gamma1": gamma1_param,
    "gamma2": gamma2_param,
    "weight": weight_param,
}

# Bayesian optimization for hyperparameter search
n_iter = 200
cv = 5
n_jobs=-1
random_state=42

opt = BayesSearchCV(
    CombinedKernelRidge(), 
    param_space, 
    optimizer_kwargs=None,
    n_iter=n_iter, 
    cv=cv, 
    n_jobs=n_jobs, 
    random_state=random_state, 
    scoring=PearsonCorrScorer(greater_is_better=True),
    refit=True,
    verbose=0,
    fit_params=None,
    error_score='raise',
    return_train_score=False
)

logging.info("Starting Bayesian optimization")
opt.fit(
    X_train, 
    y_train, 
    n_features1=n_features1, 
    n_features2=n_features2
)
logging.info("Bayesian optimization completed.")

# creat a folder
os.makedirs('results', exist_ok=True)

# Log the optimal hyperparameters
best_params = opt.best_params_
best_score = opt.best_score_
logging.info(f"Best parameters found: {best_params}")
logging.info(f"Best score found: {best_score}")

# save the optimal hyperparameters
write_params_to_file('./results/best_params.txt', best_params)

# train the model using optimal hyperparameters
best_model = CombinedKernelRidge(**best_params)
best_model.fit(
    X_train, 
    y_train, 
    n_features1=n_features1, 
    n_features2=n_features2
)

# make predictions
pred_test = best_model.predict(X_test)

# save prediction results
r = open('./example_data/val_id', 'r')
val_id = []
for i in r:
    f = i.split()
    ID = f[0]
    val_id.append(ID)
r.close()
save_predictions_to_file('./results/valID_pred.txt', val_id, pred_test)

clean_memory(X_train, y_train, X_test, val_id)
