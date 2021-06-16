sentiment_model = {}
def register_model(name):
  def decorator(cls):
    sentiment_model[name] = cls
    return cls
  return decorator
