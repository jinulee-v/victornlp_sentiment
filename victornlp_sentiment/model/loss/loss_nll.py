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
  device = next(parser.parameters()).device
  batch_size = len(inputs)
  
  scores = model.run(inputs, **kwargs)
  assert scores.size(1) == len(model.labels)
  true_labels = torch.zeros(scores.size(), device=device, dtype=torch.long).detach()

  for i, input in enumerate(inputs):
    for phrase_sent in input['sentiment']:
      true_labels[i, phrase_sent['head'], model.labels_stoi['label']] = 1
  
  loss = - torch.sum((scores * true_labels)) / batch_size

  return loss  
  