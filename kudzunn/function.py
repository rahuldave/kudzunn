import numpy as np


class Function:
    """
    An abstraction for a function. The constructor initializes the parameters
    of the function. Dunder `__call__` actually calls it, using these
    parameters as state. The gradient of the function with respect to its
    inputs and parameters must be provided using `backward`.
    """

    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, inputs):
        "Call the function"
        raise NotImplementedError

    def backward(self, grad):
        "Compute and store gradients wrt parameters, return wrt inputs"
        raise NotImplementedError

    def params_and_grads(self):
        "Obtain a list of parameter values and their gradients."
        pglist = []
        for name, param in self.params.items():
            grad = self.grads[name]
            pglist.append((name, param, grad))
        return pglist


class ZeroBiasAffine(Function):
    """
    A function that defines a linear tranform in 1-D. It multiplies the input
    by a single parameter.
    """

    def __init__(self):
        super().__init__()
        self.params["w"] = np.random.randn()
        self.grads["w"] = 0.0

    def __call__(self, inputs):
        self.inputs = inputs
        return inputs * self.params["w"]

    def backward(self, grad):
        self.grads["w"] = grad @ self.inputs
        return self.params["w"] * grad
