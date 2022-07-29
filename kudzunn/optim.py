class Optimizer:
    """
    Abstract Class for Optimizer.
    """

    def step(self, func) -> None:
        """
        Parameters
        ----------
        func: Function
            The function whose parameters need to be stepped
        """
        raise NotImplementedError


class GD(Optimizer):
    """
    Gradient Descent Optimizer.

    Parameters
    ----------
    lr: float
        The learing rate to scale the gradient with

    """

    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def step(self, func) -> None:
        """
        Makes a gradient descent step for all parameters by lr*gradient

        Parameters
        ----------
        func: Function
            The function whose parameters need to be stepped
        """
        for name, param, grad in func.params_and_grads():
            func.params[name] = param - self.lr * grad
