class Scheduler(object):
  _scheduler_list = {}
  
  def __init__(self, precisions):
    self.precisions = set(precisions)
  
  
  def __init_subclass__(cls, scheduler_name):
    super().__init_subclass__()
    cls._scheduler_list[scheduler_name] = cls
  
  
  def __class_getitem__(cls, scheduler_name):
    return cls._scheduler_list[scheduler_name]
  

  def update_precision(self, **kwargs):
    '''
    Update the precision of the model to the given precision.
    '''
    pass
  
  def reset(self):
    '''
    Reset the scheduler.
    '''
    pass
  
  
  @classmethod
  def get_scheduler(cls, scheduler_name, **kwargs):
    scheduler_cls = Scheduler[scheduler_name]
    precisions = kwargs['precisions']
    if scheduler_name == 'naive':
      high_bit_steps = kwargs['high_bit_steps']
      return scheduler_cls(precisions, high_bit_steps)
    elif scheduler_name == 'kv_cache':
      precision_switch_points = kwargs['precision_switch_points']
      dim = kwargs['dim']
      num_heads = kwargs['num_heads']
      save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
      dropout_prob = kwargs['dropout_prob'] if 'dropout_prob' in kwargs else 0.1
      max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else 255
      return scheduler_cls(precisions, precision_switch_points, dim, num_heads, save_dir=save_dir, dropout_prob=dropout_prob, max_new_tokens=max_new_tokens)
    elif scheduler_cls == 'act':
      precision_switch_points = kwargs['precision_switch_points']
      dim = kwargs['act_dim']
      save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
      max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else 255
      all_layers = kwargs['all_layers']
      return scheduler_cls(precisions, 
                           precision_switch_points, 
                           dim, 
                           save_dir=save_dir, 
                           max_new_tokens=max_new_tokens,
                           all_layers=all_layers)
    elif scheduler_name == 'confidence':
      return scheduler_cls(precisions)
    elif scheduler_name == 'random':
      p = kwargs['random_p'] if 'random_p' in kwargs else None
      max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else 255
      return scheduler_cls(precisions, p, max_new_tokens)
    else:
      raise ValueError(f"Scheduler {scheduler_name} not found.")