import numpy as np
from numpy import ndarray
from typing import Dict


class Function:
    """
    An abstraction for a function. The constructor initializes the parameters
    of the function. Dunder `__call__` actually calls it, using these
    parameters as state. The gradient of the function with respect to its
    inputs and parameters must be provided using `backward`.
    """

    def __init__(self) -> None:
        self.params: Dict[str, float] = {}
        self.grads: Dict[str, float] = {}

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Call the function. You can use the instances state in this call.

        Parameters
        ----------
        inputs: ndarray
            The inputs at which the function is called.

        Returns
        -------

        output: ndarray
            A numpy array representing output of the function call.
        """
        raise NotImplementedError

    def backward(self, grad: ndarray) -> ndarray:
        """
        Compute and store gradients wrt parameters.
        Return gradient wrt inputs.

        Parameters
        ----------
        grad: ndarray
            Gradient of the loss with respect to this function

        Returns
        -------

        outgrads: ndarray
            A numpy array representing gradient of the loss with
            respect to the inputs of this function
        """
        raise NotImplementedError

    def params_and_grads(self):
        """
        Obtain a list of parameter values and their gradients.

        Returns
        -------

        pglist: List
            A python list of tuples. Each tuple has the parameter name,
            the parameter values, and the value of the gradient wrt
            the parameter.
        """
        pglist = []
        for name, param in self.params.items():
            grad = self.grads[name]
            pglist.append((name, param, grad))
        return pglist


class ZeroBiasAffine(Function):
    """
    A function that defines a linear tranform in 1-D. It multiplies the input
    by a single parameter.

    Parameters
    ----------
    winit: float
        An initialization to provide the function parameter w.
    wgrad: float
        An initialization to provide the gradient dJ/dw.

    """

    def __init__(self, winit=None, wgrad=None) -> None:
        super().__init__()
        if winit:
            self.params["w"] = winit
        else:
            self.params["w"] = np.random.randn()
        if wgrad:
            self.grads["w"] = wgrad
        else:
            self.grads["w"] = 0.0

    def __call__(self, inputs: ndarray) -> ndarray:
        """
        Call w*x.

        Parameters
        ----------
        inputs: ndarray
            The inputs at which the function is called.

        Returns
        -------

        output: ndarray
            A numpy array representing output of the function call.
        """
        self.inputs = inputs
        return inputs * self.params["w"]

    def backward(self, grad: ndarray) -> ndarray:
        """
        Compute and store gradients wrt parameters.
        Since dJ/dw = dJ/df*df/dw we store the dot of the incoming
        gradient with inputs. Since dJ/dx = dJ/df*df/dx
        we return gradient wrt inputs as grad*w.

        Parameters
        ----------
        grad: ndarray
            Gradient of the loss with respect to this function

        Returns
        -------

        outgrads: ndarray
            A numpy array representing gradient of the loss with
            respect to the inputs of this function.
        """
        self.grads["w"] = grad @ self.inputs
        return self.params["w"] * grad
