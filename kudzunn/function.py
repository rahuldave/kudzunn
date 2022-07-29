import numpy as np


class Function:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params_and_grads(self):
        pglist = []
        for name, param in self.params.items():
            grad = self.grads[name]
            pglist.append((name, param, grad))
        return pglist


class ZeroBiasAffine(Function):
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
