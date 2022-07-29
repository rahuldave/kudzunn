import numpy as np
from kudzunn.loss import MSE


def test_loss_value():
    preds = np.array([1.1, 1.1])
    actuals = np.array([1.0, 1.0])
    el = MSE()
    lval = el(preds, actuals)
    assert np.isclose(lval, 0.01)


def test_loss_grad():
    preds = np.array([1.1, 1.1])
    actuals = np.array([1.0, 1.0])
    el = MSE()
    grads = el.backward(preds, actuals)
    assert np.allclose(grads, np.array([0.1, 0.1]))
