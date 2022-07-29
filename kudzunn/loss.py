import numpy as np


class Loss:
    "Base class for losses"

    def __call__(self, predicted, actual):
        "How the loss is called given the predictions and the ys"
        raise NotImplementedError

    def backward(self, predicted, actual):
        "Gradient of the loss with respect to the prediction function"
        raise NotImplementedError


class MSE(Loss):
    "MSE loss"

    def __call__(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted, actual):
        N = actual.shape[0]
        return (2.0 / N) * (predicted - actual)
