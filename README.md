# kudzunn

[![Build Status](https://github.com/rahuldave/kudzunn/workflows/Build%20Main/badge.svg)](https://github.com/rahuldave/kudzunn/actions)
[![Documentation](https://github.com/rahuldave/kudzunn/workflows/Documentation/badge.svg)](https://rahuldave.github.io/kudzunn/)
[![Code Coverage](https://codecov.io/gh/rahuldave/kudzunn/branch/main/graph/badge.svg)](https://codecov.io/gh/rahuldave/kudzunn)

Neural Network library for Learning

---

## Features

- Support for arbitrary losses
- You need to supply layer gradients

## Quick Start

```python
class Config:
  pass

config = Config()
config.lr = 0.01
config.num_epochs = 200
data = Data(x, y)
loss = MSE()
fn = ZeroBiasAffine()
opt = GD(config.lr)
learner = Learner(opt, loss, fn, config.num_epochs)
acc = AccCallback(learner)
learner.set_callbacks([acc])
learner.train_loop(data)
```

## Installation

**Stable Release:** `pip install kudzunn`<br>
**Development Head:** `pip install git+https://github.com/rahuldave/kudzunn.git`

## Documentation

For full package documentation please visit [rahuldave.github.io/kudzunn](https://rahuldave.github.io/kudzunn).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Four Commands You Need To Know to develop

1. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development
    dependencies (i.e. `tox`).

2. `make build`

    This will run `tox` which will run all your tests in both Python 3.7
    and Python 3.8 as well as linting your code.

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean environment.

4. `make docs`

    This will generate and launch a web browser to view the most up-to-date
    documentation for your Python package.




**MIT license**