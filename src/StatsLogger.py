from logging import Logger


class StatsLogger:
    '''
    Logger for printing in a progress bar
    '''

    def __init__(self, logger: Logger):
        # train logger
        self.train_loss = 0
        self.train_loss_sum = 0
        self.lr = 0
        self.pad_percent = 0
        self.pad_percent_sum = 0
        self.grad_norm = 0
        self.grad_norm_sum = 0
        self.duration = 0
        self.tokens_per_sec = 0
        # valid logger
        self.valid_loss = 0
        # logger
        self.logger = logger
        self.curr_epoch = -1

    def __iter__(self):
        dictionary = {
            'train_loss': f'{self.train_loss:.4f}',
            'lr': f'{self.lr:.8f}',
            'pad_percent': f'{self.pad_percent*100:.2f}%',
            'grad_norm': f'{self.grad_norm:.4f}',
        }
        if self.duration:
            dictionary['duration'] = f'{self.duration:.4f}s'
        if self.tokens_per_sec:
            dictionary['token/sec'] = f'{int(self.tokens_per_sec)}'
        yield from dictionary.items()

    def __str__(self):
        return ' | '.join(f'{k} {v}' for k, v in dict(self).items())

    def zero_stats(self, curr_epoch: int):
        self.__init__(self.logger)
        self.curr_epoch = curr_epoch

    def step(
        self,
        train_loss: float = None,
        lr: float = None,
        pad_percent: float = None,
        grad_norm: float = None,
        duration: float = None,
        tokens_per_sec: float = None,
        valid_loss: float = None,
    ):
        if train_loss:
            self.train_loss = train_loss
            self.train_loss_sum += train_loss
        self.lr = lr
        if pad_percent:
            self.pad_percent = pad_percent
            self.pad_percent_sum += pad_percent
        if grad_norm:
            self.grad_norm = grad_norm
            self.grad_norm_sum += grad_norm
        self.duration = duration
        self.tokens_per_sec = tokens_per_sec
        self.valid_loss = valid_loss

    def log_train(self):
        self.logger.info(f'Epoch {self.curr_epoch} | {str(self)}')

    def log_valid(self, best_valid_loss: float):
        log_str = f'Epoch {self.curr_epoch} | valid_loss {self.valid_loss:.4f}'
        if self.valid_loss < best_valid_loss:
            log_str += ' <------------------------------ best validation loss'
        self.logger.info(log_str)
