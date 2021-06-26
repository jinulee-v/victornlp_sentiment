"""
@module run_argmax

Simple run function that implements argmax choice.
"""

import torch

from . import register_run_fn

@register_run_fn('argmax')
def run_argmax(model, inputs, config, **kwargs):
  """
  @param model Model object. Refer to '.*' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  @param config Dictionary config file accessed with 'run' key.
  @param **kwargs Arguments to pass to the model.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  scores = model.run(inputs, **kwargs)
  best_label = torch.argmax(scores, 2).cpu()

  for input, label in zip(inputs, best_label):
    label = label[1:input['word_count']+1]
    if 'sentiment' in input:
      key = 'sentiment_predict'
    else:
      key = 'sentiment'
    input[key] = [{'head': id+1, 'label': model.labels[label_id]} for id, label_id in enumerate(label)]
  
  return input