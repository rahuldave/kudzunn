import numpy as np
from numpy import ndarray


class Loss:
    """
    Abstract Base Class for losses. Defines an interface in which
    dunder __call__ is used to compute the value of the loss (forward)
    and the `backward` method is used to compute the gradient with respect
    to the prediction function computed at the input data points.

    """

    def __call__(self, predicted: ndarray, actual: ndarray) -> float:
        """
        How the loss is called given the predictions and the ys.

        Parameters
        ----------
        predicted: ndarray
            An array of predictions of the dependent variable
        actual: ndarray
            An array of actual values of the dependent variable

        Returns
        -------

        loss: float
            The current value of the loss

        """
        raise NotImplementedError

    def backward(self, predicted: ndarray, actual: ndarray) -> ndarray:
        """
        Gradient of the loss with respect to the prediction function.

        Parameters
        ----------
        predicted: ndarray
            An array of predictions of the dependent variable
        actual: ndarray
            An array of actual values of the dependent variable

        Returns
        -------

        grads: ndarray
            The gradient of the loss with respect to the function.
        """
        raise NotImplementedError


class MSE(Loss):
    """
    MSE loss. Computes the square of the residual, sums it up over all
    points and divides by the number of points.
    """

    def __call__(self, predicted: ndarray, actual: ndarray) -> float:
        """
        Parameters
        ----------
        predicted: ndarray
            An array of predictions of the dependent variable
        actual: ndarray
            An array of actual values of the dependent variable


        Returns
        -------

        loss: float
            The current value of the loss

        """
        return np.mean((predicted - actual) ** 2)

    def backward(self, predicted: ndarray, actual: ndarray) -> ndarray:
        """
        Parameters
        ----------
        predicted: ndarray
            An array of predictions of the dependent variable
        actual: ndarray
            An array of actual values of the dependent variable

        Returns
        -------

        grads: ndarray
            The gradient of the loss with respect to the function. For
            the squared error this loss is 2/N *(residual)
        """
        N = actual.shape[0]
        return (2.0 / N) * (predicted - actual)
