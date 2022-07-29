class Optimizer:
    "Abstract Class for Optimizer"

    def step(self, func):
        raise NotImplementedError


class GD(Optimizer):
    "Gradient Descent Optimizer."

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, func):
        "Makes a gradient descent step for all parameters by lr*gradient"
        for name, param, grad in func.params_and_grads():
            func.params[name] = param - self.lr * grad
