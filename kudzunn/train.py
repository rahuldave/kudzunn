class Learner:
    def __init__(self, opt, loss, func, epochs):
        self.loss = loss
        self.func = func
        self.opt = opt
        self.epochs = epochs
        self.cbs = []

    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)

    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status

    def train_loop(self, data):
        self("fit_start")
        for epoch in range(self.epochs):
            self("epoch_start", epoch)
            inputs, targets = data[:]

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
