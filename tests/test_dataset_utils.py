import numpy as np
import pytest

from data.dataset_utils import split_dataset


def test_split_dataset_preserves_pairs_and_split_sizes(monkeypatch):
    X = np.arange(22).reshape(11, 2)
    Y = np.arange(11).reshape(11, 1)

    #A known order makes it clear which samples should be in each split
    monkeypatch.setattr(np.random, "permutation", lambda length: np.arange(length))

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    assert X_train.shape == (8, 2)
    assert Y_train.shape == (8, 1)
    assert X_val.shape == (1, 2)
    assert Y_val.shape == (1, 1)
    assert X_test.shape == (2, 2)
    assert Y_test.shape == (2, 1)

    assert np.array_equal(X_train, X[:8])
    assert np.array_equal(Y_train, Y[:8])
    assert np.array_equal(X_val, X[8:9])
    assert np.array_equal(Y_val, Y[8:9])
    assert np.array_equal(X_test, X[9:])
    assert np.array_equal(Y_test, Y[9:])


def test_split_dataset_requires_matching_sample_counts():
    with pytest.raises(ValueError):
        split_dataset(np.zeros((3, 2)), np.zeros((2, 1)))
