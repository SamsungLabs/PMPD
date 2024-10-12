from .scheduler import Scheduler
from torch.nn import Module
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torch


class ActScheduler(Scheduler, scheduler_name='act'):
    def __init__(self, precisions, precision_switch_points, dim,  save_dir=None, max_new_tokens=255, all_layers=False):
        super().__init__(precisions)
        self.classifiers = {}
        self.window_size = 3
        self.past_scores = defaultdict(list)
        self.threshold = defaultdict(float)
        self.high_bit_steps = None
        self.k = 2
        self.high_prec_steps = [0, max_new_tokens // 3, max_new_tokens // 3 * 2, max_new_tokens]
        self.all_layers = all_layers
        for (high_precision, low_precision) in precision_switch_points:
          if save_dir is None:
            self.classifiers[(high_precision, low_precision)] = ActClassifier(dim, self.all_layers)
          else:
            self.classifiers[(high_precision, low_precision)] = ActClassifier(dim, self.all_layers)
            self.classifiers[(high_precision, low_precision)].load_state_dict(torch.load(f'{save_dir}/classifier_{high_precision}_{low_precision}.pt'))
            print('Loaded classifier for precision', high_precision, low_precision)
            self.classifiers[(high_precision, low_precision)].eval()        

    def schedule(self, **kwargs):
        '''
        Schedule the precision based on activations.
        '''
        acts = kwargs['acts']

        current_precision = kwargs['precision']
        index = kwargs['index']
        if (current_precision, current_precision - 1) in self.classifiers:
          classifier = self.classifiers[(current_precision, current_precision - 1)]
          if self.high_bit_steps is None:
            with torch.no_grad():
              score = classifier(acts)
            self.high_bit_steps = self.high_prec_steps[score[0].argmax().item()]
            print('Score:', score.cpu().tolist(), 'High bit steps:', self.high_bit_steps)
          if index >= self.high_bit_steps:
            return current_precision - 1
        return current_precision
      
      
    def reset(self):
        '''
        Reset the scheduler.
        '''
        self.past_scores = defaultdict(list)
        self.threshold = defaultdict(float)
        self.high_bit_steps = None

class ActClassifier(Module):
  def __init__(self, dim, all_layers, classes=4):
    super().__init__()
    self.all_layers = all_layers
    self.project = nn.Sequential(
      nn.Linear(dim, (dim+classes) // 2),
      nn.ReLU(inplace=True),
      nn.Linear((dim+classes) // 2, classes),
    )

  def forward(self, act):
    act = act.to(torch.float32)

    if not self.all_layers:
       act = act.mean(1).squeeze()

    ret = torch.softmax(self.project(act.unsqueeze(0)), dim=-1)
    return ret  
