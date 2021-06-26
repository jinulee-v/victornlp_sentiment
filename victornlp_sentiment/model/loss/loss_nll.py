"""
@module loss_nll

Negative Log-likelihood loss function for sentiment analysis.
"""

import torch

from . import register_loss_fn

@register_loss_fn('nll')
def loss_nll(model, inputs, **kwargs):
  """
  Negative log-likelihood loss function.

  @param model Model object. Refer to '..model' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  @param **kwargs Arguments to pass to the model.
  
  @return loss loss value.
  """
  torch.autograd.set_detect_anomaly(True)
  device = next(model.parameters()).device
  batch_size = len(inputs)
  
  scores = model.run(inputs, **kwargs)
  assert scores.size(2) == len(model.labels)
  true_labels = torch.zeros_like(scores).detach()

  count = 0
  for i, input in enumerate(inputs):
    for phrase_sent in input['sentiment']:
      true_labels[i, phrase_sent['head'], model.labels_stoi[phrase_sent['label']]] = 1
      count += 1
  
  loss = - torch.sum(scores * true_labels) / count

  return loss  
  