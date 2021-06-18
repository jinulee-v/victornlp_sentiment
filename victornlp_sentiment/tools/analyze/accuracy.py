"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

@register_analysis_fn('accuracy')
def analyze_accuracy(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Accuracy value.
  """

  total = 0
  correct = 0
  for input in inputs:
    assert 'sentiment' in inputs
    assert 'sentiment_predict' in inputs

    for correct in input['sentiment']:
      total += 1
      for guess in input['sentiment_predict']:
        if correct['head'] == guess['head']:
          correct += 1
        if correct['head'] <= guess['head']:
          break
  
  return round(correct/total*100, 2)
