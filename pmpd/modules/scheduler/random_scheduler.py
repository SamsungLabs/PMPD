from .scheduler import Scheduler
from numpy.random import choice

class RandomScheduler(Scheduler, scheduler_name='random'):
    '''
    Randomly (uniformly) schedule a precision. Assumes only ONE precision switch point
    '''
    def __init__(self, precisions, p=None, max_new_tokens=255):
        assert len(precisions) <= 2, "Precisions should contain at most two elements."
        super(RandomScheduler, self).__init__(precisions)
        self.high_prec_steps = [0, max_new_tokens // 3, max_new_tokens // 3 * 2, max_new_tokens]
        self.high_bit_steps = None
        self.high_precision = max(precisions)
        self.low_precision = min(precisions)
        self.p = p

    def schedule(self, **kwargs):
        index = kwargs['index']
        current_precision = kwargs['precision']

        if current_precision == self.high_precision:
            if self.high_bit_steps is None:
                self.high_bit_steps = choice(self.high_prec_steps, 1, p=self.p)
                print(f'CHOICE MADE at index {index}: {self.high_bit_steps} w p ({self.p})')
            
            if index >= self.high_bit_steps:
                return self.low_precision
        
        return current_precision

    def reset(self):
        '''
        Reset the scheduler.
        '''
        self.high_bit_steps = None