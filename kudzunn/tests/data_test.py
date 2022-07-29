import numpy as np
from kudzunn.data import Data


def test_data_length():
    x = np.zeros(10)
    y = np.zeros(10)
    d = Data(x, y)
    assert len(d) == 10


def test_data_getitem():
    x = np.arange(10, dtype=int)
    y = np.arange(10, dtype=int)
    d = Data(x, y)
    for i in range(10):
        assert d[i] == (i, i)
