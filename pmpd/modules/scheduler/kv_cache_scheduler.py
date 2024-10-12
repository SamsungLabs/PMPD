from .scheduler import Scheduler
from torch.nn import Module
from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
import torch


class KVCacheScheduler(Scheduler, scheduler_name='kv_cache'):
    def __init__(self, precisions, precision_switch_points, dim, num_heads, save_dir=None, dropout_prob=0.1, max_new_tokens=255):
        super().__init__(precisions)
        self.classifiers = {}
        self.window_size = 3
        self.past_scores = defaultdict(list)
        self.threshold = defaultdict(float)
        self.high_bit_steps = None
        self.k = 2
        self.high_prec_steps = [0, max_new_tokens // 3, max_new_tokens // 3 * 2, max_new_tokens]
        for (high_precision, low_precision) in precision_switch_points:
          if save_dir is None:
            self.classifiers[(high_precision, low_precision)] = KVCacheClassifier(dim, num_heads, dropout_prob=dropout_prob)
          else:
            self.classifiers[(high_precision, low_precision)] = KVCacheClassifier(dim, num_heads)
            state_dict = torch.load(f'{save_dir}/classifier_{high_precision}_{low_precision}.pt')
            # filter out the keys that are not in the model
            state_dict = {k: v for k, v in state_dict.items() if k in self.classifiers[(high_precision, low_precision)].state_dict()}
            self.classifiers[(high_precision, low_precision)].load_state_dict(state_dict)
            print('Loaded classifier for precision', high_precision, low_precision)
            self.classifiers[(high_precision, low_precision)].eval()
        

    def schedule(self, **kwargs):
        '''
        Schedule the precision based on the key-value cache.
        '''
        past_key_values = kwargs['past_key_values'] 
        key = past_key_values[0]
        value = past_key_values[1]
        current_precision = kwargs['precision']
        index = kwargs['index']
        if (current_precision, current_precision - 1) in self.classifiers:
          classifier = self.classifiers[(current_precision, current_precision - 1)]
          if self.high_bit_steps is None:
            with torch.no_grad():
              score = classifier(key, value)
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
        

class ResNet(Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    return x + self.relu(self.fc(x))


class KVCacheClassifier(Module):
  def __init__(self, dim, num_heads, dropout_prob=0.1, classes=4):
    super().__init__()
    self.query = nn.Parameter(torch.empty(num_heads, num_heads, dim))
    nn.init.xavier_normal_(self.query)
    self.query.requiresGrad = True
    hidden_dim = dim * num_heads
    self.project = nn.Sequential(
      # *([ResNet(hidden_dim, hidden_dim), nn.Dropout(dropout_prob)] * 3),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, classes)
    )
    self.threshold = 0.5
    self.ln = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout_prob)


  def forward(self, key, value):
    '''
    Forward pass of the classifier.
    '''
    # Debugging: Print shapes of key, value, and query
    B, H, T, D = key.shape
    
    # Convert key and value to the same dtype as query
    key = key.to(torch.float32).mean(dim=1)
    value = value.to(torch.float32).mean(dim=1)
    # mean_k = key.mean(dim=1)
    # mean_v = value.mean(dim=1)
    # var_k = key.var(dim=1)
    # var_v = value.var(dim=1)
    # inp = torch.cat([mean_k, mean_v, var_k, var_v], dim=-1)
    
    # project = self.project.to(inp.device)
    # logit = project(inp)
    # ret = torch.softmax(logit, dim=-1)
    # return ret

    # Expand query to match the batch size and add a dimension for time
    query = self.query.expand(B, -1, -1, -1).to(key.device)

    # Compute attention scores
    att = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / (D ** 0.5))
    
    # Apply softmax to get attention weights
    softmax = F.softmax(att, dim=-1)
    
    # Apply dropout 
    softmax = self.dropout(softmax)
    # print out average attention weights and variance
    print(softmax.mean().item(), softmax.var().item())
    
    # Compute the attended values
    attended_value = torch.matmul(softmax, value).mean(dim=2)

    # Reshape the attended values
    attended_value = attended_value.transpose(-2, -1).contiguous().view(B, -1)

    # Project the attended values to x
    project = self.project.to(attended_value.device)
    ln = self.ln.to(attended_value.device)
    logit = project(ln(attended_value))

    ret = torch.softmax(logit, dim=-1)

    return ret
  
