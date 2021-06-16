"""
@module TreeLSTMSentiment

Implements sentiment analysis using Child-sum Tree LSTM with dependency parsing results.
> Kai, T. S. et al.(2015) Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
"""

import torch
import torch.nn as nn

from . import register_model

from ...victornlp_utils.module import ChildSumTreeLSTM

@register_model('tree-lstm')
class TreeLSTMSentiment(nn.Module):
  """
  @class TreeLSTMSentiment

  Sentiment analysis with TreeLSTM
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py'' for more details.
    @param labels List of sentiment labels(strings)
    @param config Dictionary config file accessed with 'TreeLSTMSentiment' key.
    """
    super(TreeLSTMSentiment, self).__init__()

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size
    self.input_size = input_size
    self.hidden_size = config['hidden_size']

    # Model layer (Tree-LSTM Encoder)
    self.encoder = ChildSumTreeLSTM(self.input_size, self.hidden_size)
    
    # Prediction layer
    self.labels = labels  # type labels
    self.labels_stoi = {}
    for i, label in enumerate(self.labels):
      self.labels_stoi[label] = i

    self.prediction = nn.Sequential(
      nn.Linear(self.hidden_size, len(self.labels)),
      nn.LogSoftmax(dim=1)
    )
  
  def run(self, inputs):
    """
    Runs the model and obtain softmax-ed scores for each possible labels.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @return scores Tensor(batch, labels). scores[i][j] contains the log(probability) of i-th sentence in batch having j-th label.
    """
    batch_size = len(inputs)
    
    # Embedding
    embedded = []
    for embedding in self.embeddings:
      embedded.append(embedding(inputs))
    embedded = torch.cat(embedded, dim=2)
    
    # Run TreeLSTM and prediction
    results = []
    for i, input in enumerate(inputs):
      hidden_state, _ = self.encoder(input['dependency'], embedded[i], 0)
      results.append(hidden_state[0].unsqueeze(0))#  Pick only hidden state of the ROOT.and
    results = torch.cat(results, dim=0)

    # results: Tensor(batch_size, hidden_size)
    results = self.prediction(results)
    # results: Tensor(batch_size, len(labels))

    return results
