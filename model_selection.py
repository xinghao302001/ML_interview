import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y."
    assert 0.0 <= test_ratio <= 1.0, "test_ratio must be valid."

    if seed:
        np.random.seed(seed=seed)
    
    shuffled_idx = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_idx = shuffled_idx[:test_size]
    train_idx = shuffled_idx[test_size:]

    X_train = [train_idx]
    y_train = [train_idx]
    X_test = [test_idx]
    y_test = [test_idx]

    return X_train, y_train, X_test, y_test

