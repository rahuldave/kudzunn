from collections import defaultdict
import numpy as np
from kudzunn.train import Learner
from typing import List, Dict


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

    def __init__(self, learner: Learner) -> None:
        self.learner = learner

    def fit_start(self) -> bool:
        return True

    def fit_end(self) -> bool:
        return True

    def epoch_start(self, epoch: int) -> bool:
        return True

    def batch_start(self, current_batch: int) -> bool:
        return True

    def after_loss(self, loss: float) -> bool:
        return True

    def batch_end(self) -> bool:
        return True

    def epoch_end(self) -> bool:
        return True


class AccCallback(Callback):
    """
    An accumulator callback accumulates parameter history and gradient history
    for every parameter, as well as the entire history of losses over the
    training run.
    """

    def __init__(self, learner: Learner) -> None:
        "Sets up history lists for each parameter, and for the losses"
        super().__init__(learner)
        self.losses: List[float] = []
        self.batch_losses: List[float] = []
        self.paramhist: Dict[str, List[float]] = defaultdict(list)
        self.gradhist: Dict[str, List[float]] = defaultdict(list)

    def fit_start(self) -> bool:
        return True

    def fit_end(self) -> bool:
        return True

    def epoch_start(self, epoch) -> bool:
        self.epoch = epoch
        self.batch_counter = 0
        return True

    def batch_start(self, current_batch) -> bool:
        self.batch = current_batch
        return True

    def after_loss(self, loss) -> bool:
        self.loss = loss
        return True

    def batch_end(self) -> bool:
        self.batch_losses.append(self.loss)
        self.batch_counter += 1
        return True

    def epoch_end(self) -> bool:
        "Display the epoch and the loss, and the parameter values. Accumulate."
        for name, fnval, grval in self.learner.func.params_and_grads():
            # Note this print does not scale to millions of parameters
            print(f"{name}, {fnval}, {grval}\n---")
            self.paramhist[name].append(fnval)
            self.gradhist[name].append(grval)
        avloss = np.mean(self.batch_losses[-(self.batch_counter + 1) :])
        print(f"Epoch {self.epoch}:\nLoss {avloss}")
        self.losses.append(avloss)
        return True
