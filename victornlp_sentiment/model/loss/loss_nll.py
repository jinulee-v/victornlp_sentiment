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
  true_labels = torch.zeros(batch_size, device=device, dtype=torch.long)

  for i, input in enumerate(inputs):
    true_labels[i] = model.labels_stoi[input['sentiment']]
  true_labels.unsqueeze(1)
  
  loss = - torch.sum(torch.gather(scores, 1, true_labels)) / batch_size

  return loss  
  