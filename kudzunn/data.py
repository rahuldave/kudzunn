import numpy as np


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(self)
        # Start an array index for later
        self.starts = np.arange(0, self.length)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]
