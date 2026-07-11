import numpy as np


def split_dataset(X, Y):
    if len(X) != len(Y):
        raise ValueError("X and Y must contain the same number of samples")

    #random shuffle of indices, while keeping each X and Y pair together
    indices = np.random.permutation(len(X))

    #80% of data goes to training, 10% to validation and 10% to testing
    #For 50000 samples this gives 40000 training, 5000 validation and 5000 test
    train_end = int(0.8 * len(X))
    val_end = int(0.9 * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
