import torch
import torch.nn as nn

import time

class LayerNorm(nn.Module):
    def __init__(self, size, epsilon=1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        #print(mean)
        #time.sleep(10)
        return self.alpha * (x - mean)/(std + self.epsilon) + self.beta




class TestLayerNorm:
    def __init__(self) -> None:
        print("Testing Layer norm implementation against Pytorch version")

    def get_torch(self, x):
        layer_norm = nn.LayerNorm(x.shape)
        return layer_norm(x)
    
    def get_above_impl(self, x):
        layer_norm = LayerNorm(x.shape)
        return layer_norm.forward(x)

if __name__ == '__main__':
    test = TestLayerNorm()
    batch, sentence_length, embedding_dim = 20, 5, 10
    torch.manual_seed(0)
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    print(test.get_torch(embedding).shape)
    print("============================================================================")
    print(test.get_above_impl(embedding).shape)
    # Looks like the difference could be the params initialization, NEED TO CHECK MORE
