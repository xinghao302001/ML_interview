import numpy as np
from utils.metrics import r2_score


class LinearRegressor:
    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def fit_normal(self, X_train, y_train):
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b.T).dot(y_train))

        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        return self

    ###======= fit implemented by batch-gd ===========#####
    def fit_bgd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta) ** 2)) / len(y)
            except:
                return float("inf")

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            # for theta_0
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, init_theta, eta, n_iters=1e2, epsilon=0.01):
            assert (
                X_train.shape[0] == y_train.shape[0]
            ), "the size of X_train must be equal to the size of y_train"
            theta = init_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones(len(X_train), 1), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta_ = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    ###======= fit implemented by stochastic-gd ===========#####
    def fit_sgd(self, X_train, y_train, eta=0.01, n_iters=1e3, t0=5, t1=50):
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.0

        def sgd(X_b, y, init_theta, n_iters=5, t0=5, t1=50):
            def learning_rate(t):
                return t0 / (t + t1)

            theta = init_theta
            m = len(X_b)
            for step in range(n_iters):
                idxes = np.random.permutation(m)
                X_b_new = X_b[idxes, :]
                y_new = y[idxes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(step * m + step) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train)), 1), X_train])
        theta_init = np.random.randn(X_b.shape[1])
        self.theta_ = sgd(X_b, y_train, theta_init, n_iters, t0, t1)
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict(self, X_pred):
        assert (
            self.intercept_ is not None and self.coef_ is not None
        ), "must fit before fit"
        assert X_pred.shape[1] == len(
            self.coef_
        ), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones(len(X_pred), 1), X_pred])
        return X_b.dot(self.theta_)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return r2_score(y_pred, y_test)

    def __repr__(self):
        return "LinearRegressor()"
