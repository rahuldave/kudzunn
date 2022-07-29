import numpy as np
from numpy import ndarray
from typing import Tuple


class Data:
    """
    A sequence abstraction that represents data.

    Parameters
    ----------
    x: ndarray
        Independent Variable in 1D
    actual: ndarray
        Dependent variable in 1D
    shuffle: bool
        Should we shuffle the data?
    """

    def __init__(self, x: ndarray, y: ndarray, shuffle: bool = True) -> None:
        self.x = x
        self.y = y
        self.length = len(self)
        # Start an array index for later
        self.starts = np.arange(0, self.length)

    def shuffle(self) -> Tuple[ndarray, ndarray]:
        """
        Shuffle the data.

        Returns
        -------
        shuffled: (ndarray, ndarray)
            A tuple of shuffled ndarrays, x first, y second
        """
        if self.shuffle:
            np.random.shuffle(self.starts)
        return self.x[self.starts], self.y[self.starts]

    def __len__(self) -> int:
        """
        Get length of the data.

        Returns
        -------
        length: int
            The length of the y (and thus x) arrays
        """
        return self.y.shape[0]

    def __getitem__(self, i: int) -> Tuple[float, float]:
        """
        Get a pair of x,y at an index

        Returns
        -------
        xy: (float, float)
            The (x, y) tuple at an index i
        """
        return self.x[i], self.y[i]
