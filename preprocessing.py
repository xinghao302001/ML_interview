import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        
        assert X.ndim == 2, "The dimension of X must be 2."
        self.mean_  = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.std_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self
    
    def transform(self, X):
        assert X.dim == 2, "The dimension of X must be 2."
        assert self.mean_ is not None and self.std_ is not None, "must fit before transform."
        assert X.shape[1] == len(self.mean_), "the feature number of X must be equal to mean_ and std_."

        res = np.empty(shape=X.shape, dtype=float)

        for col in range(res(X.shape[1])):
            res[:, col] = (X[:,col] - self.mean_[col]) / self.std_[col]
        
        return res

        