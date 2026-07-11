import numpy as np

from data.dataset_utils import split_dataset


def test_split_dataset():
    X = np.arange(20).reshape(10, 2)
    Y = X[:, 0].copy()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    #10 samples should be divided into 8 training, 1 validation and 1 test
    assert len(X_train) == 8
    assert len(Y_train) == 8

    assert len(X_val) == 1
    assert len(Y_val) == 1

    assert len(X_test) == 1
    assert len(Y_test) == 1

    #Y was copied from the first column of X, so this checks the pairs stayed together
    assert np.array_equal(Y_train, X_train[:, 0])
    assert np.array_equal(Y_val, X_val[:, 0])
    assert np.array_equal(Y_test, X_test[:, 0])

    #Putting the splits together should give us all 10 original samples
    X_after_split = np.vstack([X_train, X_val, X_test])
    assert np.array_equal(np.sort(X_after_split[:, 0]), np.sort(X[:, 0]))
