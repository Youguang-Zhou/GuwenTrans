import numpy as np


class StatsLogger:
    '''
    Logger for printing in a progress bar
    '''

    def __init__(self):
        self.train_loss_arr = []
        self.lr = 0

    @property
    def train_loss(self):
        return np.mean(self.train_loss_arr)

    def __iter__(self):
        yield from {
            'train_loss': f'{self.train_loss:.4f}',
            'lr': f'{self.lr:.8g}',
        }.items()

    def __str__(self):
        return ' | '.join(f'{k} {v}' for k, v in dict(self).items())

    def step(self, loss, lr):
        self.train_loss_arr.append(loss)
        self.lr = lr

    def zero_stats(self):
        self.__init__()
