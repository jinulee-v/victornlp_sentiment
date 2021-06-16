"""
@module loss
Various loss functions for dependency parsing.

loss_*(parser, inputs)
  @param parser *Parser object. Refer to 'model.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  
  @return Loss value.
"""

sentiment_loss_fn = {}
def register_loss_fn(name):
  def decorator(fn):
    sentiment_loss_fn[name] = fn
    return fn
  return decorator
