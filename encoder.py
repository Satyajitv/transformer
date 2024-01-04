import torch
import torch.nn as nn

import encoding_layer
from layer_norm import LayerNorm
from transformer_utils import clone_modules

class Encoder(nn.Module):
    
    def __init__(self, layer, N) -> None:
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clone_modules(layer, N)
        self.norm = LayerNorm(layer.size) #Norm is used to pass the encoded data into decoder

    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
        
