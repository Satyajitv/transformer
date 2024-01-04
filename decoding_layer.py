import torch
import torch.nn as nn

from transformer_utils import clone_modules
from layer_norm import LayerNorm
from sub_layer_conn import SubLayerConn

class DecodingLayer(nn.Module):

    def __init__(self, size, attention, encoder_attention, feed_forward, dropout) -> None:
        super(DecodingLayer, self).__init__()
        self.size = size
        self.attention = attention
        self.feed_forward = feed_forward
        self.encoder_attention = encoder_attention

        self.add_layer_norm_list = clone_modules(SubLayerConn(self.size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask): # check what is memory supposed to be here, think it should be the input from encoder
        m = memory
        x = self.add_layer_norm_list[0](x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.add_layer_norm_list[1](x, lambda x: self.encoder_attention(x, m, m, src_mask)) 
        return self.add_layer_norm_list[2](x, self.feed_forward)
