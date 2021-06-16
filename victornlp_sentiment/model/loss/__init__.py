"""
@module loss
Various loss functions for dependency parsing.

loss_*(model, inputs)
  @param model Model object. Refer to '..model' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Loss value.
"""

sentiment_loss_fn = {}
def register_loss_fn(name):
  def decorator(fn):
    sentiment_loss_fn[name] = fn
    return fn
  return decorator
