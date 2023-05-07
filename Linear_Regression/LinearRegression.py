import numpy as  np

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train."

        x_mean = np.array(x_train)
        y_mean = np.array(y_train)
        ### =============== implemented without vectorization ========================= ###
        # numerator = 0.0
        # denominator = 0.0 

        # for x, y in zip(x_train, y_train):
        #     numerator += (x - x_mean) * (y - y_mean)
        #     denominator += (x - x_mean) ** 2
        
        # self.a_ = numerator / denominator
        # self.b_ = y_mean - self.a_ * x_mean

        ### =============== implemented with vectorization ========================= ### 
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self
    
    def predcit(self, x_predict):
        assert x_predict.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict."

        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x):
        return self.a_ * x + self.b_
    
    def __repr__(self) -> str:
        return "SimpleLinearRegression"