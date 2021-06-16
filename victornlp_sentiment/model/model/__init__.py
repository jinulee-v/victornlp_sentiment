"""
@module model
Various sentiment analysis models.

class *Sentiment(nn.Module):
  run(model, inputs, **kwargs)
    @param model Model object. Refer to '.*' for more details.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @param **kwargs Other keywords required for each model.
"""

sentiment_model = {}
def register_model(name):
  def decorator(cls):
    sentiment_model[name] = cls
    return cls
  return decorator

from .TreeLSTMSentiment import TreeLSTMSentiment