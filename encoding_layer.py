import torch
import torch.nn as nn

from transformer_utils import clone_modules
from sub_layer_conn import SubLayerConn

class EncodingLayer(nn.Module):
    
    def __init__(self, size, attention, feed_forward, dropout) -> None:
        super(EncodingLayer, self).__init__()
        # from the architecture diagram, left portion, the encoder part contains
        # 1. Multi head attention
        # 2. Feed Forward
        # 3.Two Add & Norm residual connections
        self.attention = attention
        self.feed_forward = feed_forward
        self.size = size

        self.add_layer_norm_list = clone_modules(SubLayerConn(self.size, dropout), 2)

    def forward(self, x, mask):
        x = self.add_layer_norm_list[0](x, lambda x: self.attention(x, x, x, mask))
        return self.add_layer_norm_list[1](x, self.feed_forward)