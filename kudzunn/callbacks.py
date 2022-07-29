from collections import defaultdict


class Callback:
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
    def __init__(self, learner):
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
        print(f"Epoch {self.epoch}:\nLoss {self.loss}")
        for name, fnval, grval in self.learner.func.params_and_grads():
            print(f"{name}, {fnval}, {grval}\n---")
            self.paramhist[name].append(fnval)
            self.gradhist[name].append(grval)
        self.losses.append(self.loss)
        return True
