"""
@module parse
Various parsing functions based on attention scores.

run(model, inputs, config)
  @param model Model object. Refer to '.*' for more details.
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  @param config Dictionary config file accessed with 'run' key.
"""

sentiment_run_fn = {}
def register_parse_fn(name):
  def decorator(fn):
    sentiment_run_fn[name] = fn
    return fn
  return decorator
