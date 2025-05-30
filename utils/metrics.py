import numpy as np

def acc_score(y, y_pred):
    assert y.shape[0] == y_pred.shape[0],"the size of y_true must be equal to the size of y_predict"

    return sum(y_pred==y) / len(y)


def mean_squared_error(y_true, y_pred):
    assert len(y_true) == len(y_pred), "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true - y_pred ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_pred):
    assert len(y_true) == len(y_pred), "the size of y_true must be equal to the size of y_predict"
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    assert len(y_true) == len(y_pred), "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absolute(y_true - y_pred)) / len(y_true)
    
def r2_score(y_true, y_pred):
     return 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)