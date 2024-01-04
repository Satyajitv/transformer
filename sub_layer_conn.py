import torch
import torch.nn as nn

from layer_norm import LayerNorm

class SubLayerConn(nn.Module):
    def __init__(self, size, dropout) -> None:
        super(SubLayerConn, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalized_layer = LayerNorm(size)

    def forward(self, x, prev_layer):
        return x + self.dropout(prev_layer(self.normalized_layer(x)))
