## implemented based on LR
import numpy as np
from utils.metrics import acc_score


class LogisticRegression:
    def __init__(self) -> None:
        """initial the model of LogisticR"""
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def _sigmoid(Self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eta: float = 0.01,
        n_iters: int = 1e3,
    ):
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"

        def J(theta: float, X_b: np.ndarray, y: np.ndarray):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(
                    y_hat
                )
            except:
                return float("inf")

        def dJ(theta: float, X_b: np.ndarray, y: np.ndarray):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        def gradient_descent(X_b, y, init_theta, eta, n_iters=1e4, epsilon=1e-10):
            theta = init_theta
            step = 0

            while step < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                step += 1

            return theta

        X_b = np.hstack([np.ones(len(X_train), 1), X_train])
        init_theta = np.zeros(X_b.shape[1])
        self.theta_ = gradient_descent(X_b, y_train, init_theta, eta, n_iters)

        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def pred_prob(self, X_pred):
        assert (
            self.intercept_ is not None and self.coef_ is not None
        ), "must fit before predict!"

        assert X_pred.shape[1] == len(
            self.coef_
        ), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones(len(X_pred), 1), X_pred])
        return self._sigmoid(X_b.dot(self.theta_))

    def predict(self, X_pred):
        assert (
            self.intercept_ is not None and self.coef_ is not None
        ), "must fit before predict!"
        assert X_pred.shape[1] == len(
            self.coef_
        ), "the feature number of X_predict must be equal to X_train"

        prob = self.pred_prob(X_pred)
        return np.array(prob >= 0.5, dtype="int")

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return acc_score(y_test, y_pred)

    def __repr__(self):
        return "LogisticRegression()"
