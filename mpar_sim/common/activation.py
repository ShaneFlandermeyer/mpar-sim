#
# Author: Shane Flandermeyer
# Created on Thu Feb 16 2023
# Copyright (c) 2023
#
# This file provides numpy implementations for common activation functions.
#


import numpy as np

def softmax(x):
  num = np.exp(x)
  den = np.sum(np.exp(x))
  return num / den
  