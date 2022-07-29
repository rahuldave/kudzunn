from collections import defaultdict


class Callback:
    """
    A callback is a piece of code you want to run at a given moment in the
    lifecycle of a training loop. In kudzunn, callbacks are instances of the
    class `Callback` and its derived classes. The methods of these classes
    are run at different times. Here we support callbacks at fit start and
    end, batch start and end, epoch start and end, and finally, after the loss
    is computed in each epoch. Callback methods must return `True` on proper
    completion: this is how we signal that the next callback can be run at
    that moment. The next callback will be the same method of another callback
    instance.
    """

    def __init__(self, learner):
        self.learner = learner

    def fit_start(self):
        return True

    def fit_end(self):
        return True

    def epoch_start(self, epoch):
        return True

    def batch_start(self, batch):
        return True

    def after_loss(self, loss):
        return True

    def batch_end(self):
        return True

    def epoch_end(self):
        return True


class AccCallback(Callback):
    """
    An accumulator callback accumulates parameter history and gradient history
    for every parameter, as well as the entire history of losses over the
    training run.
    """

    def __init__(self, learner):
        "Sets up history lists for each parameter, and for the losses"
        super().__init__(learner)
        self.losses = []
        self.paramhist = defaultdict(list)
        self.gradhist = defaultdict(list)

    def fit_start(self):
        return True

    def fit_end(self):
        return True

    def epoch_start(self, epoch):
        self.epoch = epoch
        return True

    def after_loss(self, loss):
        self.loss = loss
        return True

    def epoch_end(self):
        "Display the epoch and the loss, and the parameter values. Accumulate."
        print(f"Epoch {self.epoch}:\nLoss {self.loss}")
        for name, fnval, grval in self.learner.func.params_and_grads():
            # Note this print does not scale to millions of parameters
            print(f"{name}, {fnval}, {grval}\n---")
            self.paramhist[name].append(fnval)
            self.gradhist[name].append(grval)
        self.losses.append(self.loss)
        return True
