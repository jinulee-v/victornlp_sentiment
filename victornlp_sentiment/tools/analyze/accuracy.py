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
  accurate = 0
  total_label = {}
  accurate_label = {}
  for input in inputs:
    assert 'sentiment' in input
    assert 'sentiment_predict' in input

    for correct in input['sentiment']:
      if correct['label'] not in total_label:
        total_label[correct['label']] = 0
        accurate_label[correct['label']] = 0
      
      total += 1
      total_label[correct['label']] += 1

      for guess in input['sentiment_predict']:
        if correct['head'] == guess['head']:
          if correct['label'] == guess['label']:
            accurate += 1
            accurate_label[correct['label']] += 1
        if correct['head'] <= guess['head']:
          break
  
  return {
    'total': round(accurate/total*100, 2),
    'per_label': {
      label:round(accurate_label[label]/total_label[label]*100, 2) for label in sorted(list(total_label.keys()))
    }
  }
