import numpy as np
from numpy import ndarray
from typing import Tuple, Generator, List


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

    def shuffle(self):
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

    def __getitem__(self, i: int) -> Tuple:
        """
        Get a pair of x,y at an index

        Returns
        -------
        xy: (int, int)
            The (x, y) tuple at an index i
        """
        return self.x[i], self.y[i]


# idea+implementation taken from fast.ai
class Sampler:
    def __init__(self, data: Data, bs: int, shuffle: bool = False):
        "initialize sampler which will give us a batch of shuffled indexes"
        self.n = len(data.y)
        self.idxs = np.arange(0, self.n)
        self.bs = bs
        self.shuffle = shuffle

    def __iter__(self) -> Generator[List[int], None, None]:
        "a generator for a batch size sized list of indexes"
        if self.shuffle:
            np.random.shuffle(self.idxs)
        for i in range(0, self.n, self.bs):
            yield self.idxs[i : i + self.bs]


# this dataloader uses the Sampler
class Dataloader:
    def __init__(self, data: Data, sampler: Sampler):
        self.data = data
        self.sampler = sampler
        self.current_batch = 0

    def __iter__(self):
        for idxsample in self.sampler:
            yield self.data[idxsample]
            self.current_batch += 1
