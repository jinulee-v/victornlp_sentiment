"""
@module run_argmax

Simple run function that implements argmax choice.
"""

import torch

from . import register_run_fn

@register_run_fn('argmax')
def run_argmax(model, inputs, config):
  """
  @param model Model object. Refer to '.*' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  @param config Dictionary config file accessed with 'run' key.
  """
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  scores = model.run(inputs, **kwargs)
  best_label = torch.argmax(scores, 2).cpu()

  for input, label in zip(inputs, best_label):
    if 'sentiment' in input:
      key = 'sentiment'
    else:
      key = 'sentiment_predict'
    input[key] = [model.labels[label_id] for label_id in label]

  
  return input