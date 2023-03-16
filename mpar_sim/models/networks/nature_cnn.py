from typing import Callable, Optional
import torch
import torch.nn as nn
import numpy as np


class NatureCNN(nn.Module):
  """CNN architecture from the DQN Nature paper."""

  def __init__(self,
               c: int, h: int, w: int,
               out_features: int = 512,
               layer_init: Optional[Callable] = None):
    super().__init__()
    
    if layer_init is None:
      layer_init = lambda layer : layer
    self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
    self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
    self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    
    

    x = torch.zeros(c, h, w)
    hidden_shape = self.conv3(self.conv2(self.conv1(x))).shape
    self.fc = nn.Linear(np.prod(hidden_shape), out_features)

  def forward(self, x):
    x = self.conv1(x)
    x = nn.functional.relu(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.conv3(x)
    x = nn.functional.relu(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x
