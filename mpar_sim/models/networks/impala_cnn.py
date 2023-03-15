import torch.nn as nn
import torch
import numpy as np


class ResidualBlock(nn.Module):
  """A basic two-layer residual block."""
  def __init__(self, in_channels: int) -> None:
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = nn.ReLU()(x)
    out = self.conv1(out)
    out = nn.ReLU()(out)
    out = self.conv2(out)
    return out + x


class ImpalaBlock(nn.Module):
  """
  An "impala block" as described in Espeholt et al. 2018.
  
  Contains two residual blocks and a pooled convolutional layer
  """
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super(ImpalaBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(out_channels)
    self.res2 = ResidualBlock(out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x


class ImpalaModel(nn.Module):
  """The full impala network"""
  def __init__(self, h: int, w: int, c: int) -> None:
    super(ImpalaModel, self).__init__()
    self.imp1 = ImpalaBlock(in_channels=c, out_channels=16)
    self.imp2 = ImpalaBlock(in_channels=16, out_channels=32)
    self.imp3 = ImpalaBlock(in_channels=32, out_channels=32)

    out_shape = self.imp3(
        self.imp2(self.imp1(torch.zeros(c, h, w)))).shape
    self.fc = nn.Linear(in_features=np.prod(out_shape), out_features=256)

    self.output_dim = 256

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.imp1(x)
    x = self.imp2(x)
    x = self.imp3(x)
    x = nn.ReLU()(x)
    x = nn.Flatten()(x)
    x = self.fc(x)
    x = nn.ReLU()(x)
    return x