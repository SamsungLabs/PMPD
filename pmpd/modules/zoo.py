import torch
import torch.nn as nn
from auto_gptq import AutoGPTQForCausalLM
import json

class MultiPrecModelWrapper(nn.Module):
  def __init__(self, model_dict, config=None):
    super(MultiPrecModelWrapper, self).__init__()
    self.models = nn.ModuleDict(model_dict)
    self.precisions = list([int(p) for p in self.models.keys()])
    self.config = config


  @classmethod
  def from_quantized(cls, model_path, precisions, quantize_model_cls=AutoGPTQForCausalLM):
    '''
    Load models from quantized model files.
    '''
    model_paths = json.load(open(model_path))
    model_dict = {}
    for precision in precisions:
      assert str(precision) in model_paths, f"Model for precision {precision} is not found in the config."
      model_dict[str(precision)] = quantize_model_cls.from_quantized(model_paths[str(precision)]).eval().cuda()
    return cls(model_dict, config=model_paths)


  def forward(self, x, precision, **kwargs):
    if precision < 0 or precision > max(self.precisions):
      raise ValueError("Invalid precision value")

    selected_model = self.models[str(precision)]
    output = selected_model(x, **kwargs)
    return output