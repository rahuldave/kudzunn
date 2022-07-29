from kudzunn.optim import Optimizer
from kudzunn.loss import Loss
from kudzunn.function import Function
from kudzunn.callbacks import Callback
from kudzunn.data import Data
from typing import List


class Learner:
    """
    Learner class encapsulates the training. The constructor
    initializes. We can then set the callbacks and run the training loop

    Parameters
    ----------
    opt: Optimizer
        optim to use. currently only an instance of `GD()`
    loss: Loss
        the class representing the loss function
    func: Function
        currently the function in this 1-layer network
    epochs: int
        The number of epochs to train the model
    """

    def __init__(self, opt: Optimizer, loss: Loss, func: Function, epochs: int) -> None:
        self.loss = loss
        self.func = func
        self.opt = opt
        self.epochs = epochs
        self.cbs: List[Callback] = []

    def set_callbacks(self, cblist: List[Callback]) -> None:
        """
        Take a list of callbacks and add it to the internal callback array

        Parameters
        ----------
        cblist: List[Callback]
            An list of callback class instances
        """

        for cb in cblist:
            self.cbs.append(cb)

    def __call__(self, cbname: str, *args) -> bool:
        """
        hack to use dunder call to run the same method in all callbacks

        Parameters
        ----------
        cbname: str
            a string with the name of the callback method
        args: List
            a list of other arguments of the callbacks

        Returns
        -------
        success: bool
            a boolean representing whether all the callbacks succeeded
        """
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status

    def train_loop(self, data: Data) -> float:
        """
        The training loop over epochs!

        The calculation of the loss, and then the backpropagation,
        and finally the optimizer step.

        Callbacks are run at appropriate spots.

        Parameters
        ----------
        data: Data
            an instance of the Data class supplied.

        Returns
        -------
        finalloss: float
            loss at end of all the epochs
        """
        self("fit_start")
        for epoch in range(self.epochs):
            self("epoch_start", epoch)
            inputs, targets = data.shuffle()

            # make predictions
            predicted = self.func(inputs)

            # actual loss value
            epochloss = self.loss(predicted, targets)
            self("after_loss", epochloss)

            # calculate gradient
            intermed = self.loss.backward(predicted, targets)
            self.func.backward(intermed)

            # update parameter with gradient
            self.opt.step(self.func)

            self("epoch_end")
        self("fit_end")
        return epochloss
