import torch
import torch.nn as nn

import math

from transformer_utils import clone_modules

class MultiHeadAttention(nn.Module):

    def __init__(self, heads=8, embed_size=512, dropout=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # this splits the embedding into multiple parts based on heads count,
                                            # so each head can learn something different and also helps with parallelism during training.
        
        assert(self.head_dim * heads == embed_size), "Adjust heads to make sure we split and consider the entire embed into all heads"

        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # output should be batch * head_dim=64 with embed_size=512, heads=8
        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # output should be batch * head_dim=64 with embed_size=512, heads=8
        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # output should be batch * head_dim=64 with embed_size=512, heads=8

        self.linears_QKV = clone_modules(nn.Linear(self.embed_size, self.embed_size, bias=False), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None



    def forward(self, query, key, value, mask=None):
        
        if mask is not None:
            mask = mask.unsqueeze(1) # Not sure about this one CHECK LATER

        B = query.size(0)

        # shape of x here is B, T, C will be changed into B, HEADS, T, C - (C will be split into d_model/heads)
        query, key, value = [ lin(x).view(B, -1, self.heads, self.head_dim).transpose(1, 2) for lin, x in zip(self.linears_QKV, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(B , -1, self.heads * self.head_dim)
        )
        del query
        del key
        del value

        return self.linears_QKV[-1](x)

        


def attention(Q, K, V, mask=None, dropout=None):
    # K - (B, T, C)  Batch Time and Actual embed vector of word
    # Q - same as above
    # V - same as above

    # considering one sentence VEC =  (T, C)
    # if we multiply VEC @ VEC^T we get T * C matrix where each row tells the vectors are other vectors associated closeness by dot product
    # SO we do Q @ K^T
    dim_k = K.size(-1) # Gives the dimension of the word emebedding
    # Now dot product between Q @ K^T
    K_T = K.transpose(-2, -1) # B, C, T
    #comp_scores = (Q @ K_T) / torch.sqrt(dim_k)
    comp_scores = torch.matmul(Q, K_T) / math.sqrt(dim_k)
    norm = comp_scores.softmax(dim=-1)
    if dropout is not None:
        norm = dropout(norm)

    return (norm @ V), norm


