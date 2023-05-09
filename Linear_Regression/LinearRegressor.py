import numpy as np
from metrics import r2_score


class LinearRegressor:

    def __init__(self) -> None:

        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train),1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b.T).dot(y_train))
        
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        return self
    
    def predict(self, X_pred):
        assert self.intercept_ is not None and self.coef_ is not None, "must fit before fit"
        assert X_pred.shape[1] == len(self.coef_), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones(len(X_pred),1), X_pred])
        return X_b.dot(self.theta_)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return r2_score(y_pred, y_test)
    
    def __repr__(self):
        return "LinearRegressor()"