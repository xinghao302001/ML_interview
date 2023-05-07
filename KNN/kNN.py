import numpy as np
from math import sqrt
from collections import Counter
from .metrics import acc_score


class KNNClassifier:

    def __init__(self, k):
        assert k >= 1 and isinstance(k, int) , "k is not valid."
        self.k = k
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train is not equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train is not at least k."
        
        self.X_train_ = X_train
        self.y_train_ = y_train
        return self
    
    def predict(self, X_predict):
        assert self.X_train_ is not None and self.y_train_ is not None, "must fit before predict"
        assert X_predict.shape[1] == self.X_train_.shape[1], "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict)
    
    def _predict(self, x):
        assert x.shape[0] == self.X_train_.shape[1], "the feature number of x must be equal X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train_]
        sorted_distance = np.argsort(distances)

        topK_y = [self.y_train_[i] for i in sorted_distance[:self.k]]
        votes = Counter(topK_y)
        
        return votes.most_common(1)[0][0]
        

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return acc_score(y_test, y_predict)
    

    def __repr__(self):
        return "KNN(k=%d)" % self.k