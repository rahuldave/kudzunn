import numpy as np


class Loss:
    def __call__(self, predicted, actual):
        raise NotImplementedError

    def backward(self, predicted, actual):
        raise NotImplementedError


class MSE(Loss):
    def __call__(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted, actual):
        N = actual.shape[0]
        return (2.0 / N) * (predicted - actual)
