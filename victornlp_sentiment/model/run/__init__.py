"""
@module parse
Various parsing functions based on attention scores.

run_*(parser, inputs, config)
  @param parser *Parser object. Refer to '*_parser.py' for more details.
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  @param config Dictionary config file accessed with 'parse' key.
"""

sentiment_run_fn = {}
def register_parse_fn(name):
  def decorator(fn):
    sentiment_run_fn[name] = fn
    return fn
  return decorator
