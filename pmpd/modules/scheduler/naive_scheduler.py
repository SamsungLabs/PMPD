from .scheduler import Scheduler

class NaiveScheduler(Scheduler, scheduler_name='naive'):
    '''
    Use high and low precision to schedule the tasks.
    '''
    def __init__(self, precisions, high_precision_steps):
        assert len(precisions) <= 2, "Precisions should contain at most two elements."
        super(NaiveScheduler, self).__init__(precisions)
        self.high_precision = max(precisions)
        self.low_precision = min(precisions)
        self.high_precision_steps = high_precision_steps

    def schedule(self, **kwargs):
        index = kwargs['index']
        if index < self.high_precision_steps:
            return self.high_precision
        else:
            return self.low_precision
        