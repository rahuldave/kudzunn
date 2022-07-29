import numpy as np
from kudzunn.function import ZeroBiasAffine


def test_zba_constructor():
    f = ZeroBiasAffine(winit=1, wgrad=0.2)
    assert f.params["w"] == 1
    assert f.grads["w"] == 0.2


def test_zba_value():
    f = ZeroBiasAffine(winit=1.2, wgrad=0.2)
    inputs = np.ones(3)
    assert np.allclose(f(inputs), np.array([1.2, 1.2, 1.2]))


def test_zba_grad_dx():
    f = ZeroBiasAffine(winit=1.2, wgrad=0.2)
    inputs = np.ones(3)
    f(inputs)
    incoming_grads = np.ones(3)
    dfdx = f.backward(incoming_grads)
    assert np.allclose(dfdx, np.array([1.2, 1.2, 1.2]))


def test_zba_grad_dw():
    f = ZeroBiasAffine(winit=1.2, wgrad=0.2)
    inputs = np.ones(3)
    f(inputs)
    incoming_grads = np.ones(3)
    f.backward(incoming_grads)
    assert np.isclose(f.grads["w"], 3.0)
