import numpy as np

def softmax(x):
  num = np.exp(x)
  den = np.sum(np.exp(x))
  return num / den
  