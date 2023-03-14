import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                           kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                           kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor):
    out = nn.ReLU()(x)
    out = self.conv1(out)
    out = nn.ReLU()(out)
    out = self.conv2(out)
    return out + x


class ImpalaBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(in_channels=out_channels)
    self.res2 = ResidualBlock(in_channels=out_channels)

  def forward(self, x: torch.Tensor):
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x


class ImpalaModel(nn.Module):
  def __init__(self,
               h: int,
               w: int,
               c: int):
    super().__init__()
    self.imp1 = ImpalaBlock(in_channels=c, out_channels=16)
    self.imp2 = ImpalaBlock(in_channels=16, out_channels=32)
    self.imp3 = ImpalaBlock(in_channels=32, out_channels=32)

    out_shape = self.imp3(self.imp2(self.imp1(torch.zeros(c, h, w)))).shape
    self.fc = nn.Linear(in_features=np.prod(out_shape), out_features=256)

  def forward(self, x: torch.Tensor):
    x = self.imp1(x)
    x = self.imp2(x)
    x = self.imp3(x)
    x = nn.Flatten()(x)
    x = self.fc(x)
    return x