# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from torch import nn


class StreamableLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class ContextModel(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers, batch_first=True)
        
    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out[:, -1, :]