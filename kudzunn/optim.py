class Optimizer:
    def step(self, func):
        raise NotImplementedError


class GD(Optimizer):
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, func):
        for name, param, grad in func.params_and_grads():
            func.params[name] = param - self.lr * grad
