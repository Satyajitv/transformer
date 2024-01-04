import torch
import torch.nn as nn

from transformer_utils import clone_modules
from layer_norm import LayerNorm

class Decoder(nn.Module):
    def __init__(self, layer, N) -> None:
        super(Decoder, self).__init__()
        self.layers = clone_modules(layer, N)
        self.layer_norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_emb, target_emb):
        for layer in self.layers:
            x = layer(x, memory, src_emb, target_emb)
            
        return self.layer_norm(x)