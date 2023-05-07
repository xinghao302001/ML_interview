import numpy as np

def acc_score(y, y_pred):
    assert y.shape[0] == y_pred.shape[0],"the size of y_true must be equal to the size of y_predict"

    return sum(y_pred==y) / len(y)