from .scheduler import Scheduler
from torch.nn import Module
from collections import defaultdict
import torch


class ConfidenceScheduler(Scheduler, scheduler_name='confidence'):
    def __init__(self, precisions):
        super().__init__(precisions)
        self.window_size = 10 
        self.past_scores = defaultdict(list)
        self.threshold = defaultdict(float)
        self.k = 1
        

    def schedule(self, **kwargs):
        '''
        Schedule the precision based on the confidence.
        '''
        logits = kwargs['logits']
        current_precision = kwargs['precision']
        if (current_precision - 1) in self.precisions:
          score = logits[:, -1:].softmax(dim=-1).max().item()
          # print('Score:', score)
          if len(self.past_scores[current_precision]) < self.window_size:
            self.past_scores[current_precision].append(score)
          else:
            # if score > classifier.threshold:
            if self.threshold[current_precision] == 0:
              mean = torch.tensor(self.past_scores[current_precision]).mean()
              std = torch.tensor(self.past_scores[current_precision]).std()
              self.threshold[current_precision] = mean + self.k * std
              # print('Threshold:', self.threshold)
            if score >= self.threshold[current_precision]:
              return current_precision - 1
        return current_precision
      
      
    def reset(self):
        '''
        Reset the scheduler.
        '''
        self.past_scores = defaultdict(list)
        self.threshold = defaultdict(float)
        