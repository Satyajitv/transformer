import copy

from attention import MultiHeadAttention
from positional_embedding import PositionalEncoding
from pointwise_ff import PositionwiseFeedForward
from encoder_decoder import EncoderDecoder, Generator
from encoder import Encoder
from decoder import Decoder
from encoding_layer import EncodingLayer
from decoding_layer import DecodingLayer
from embeddings import Embeddings

import torch
import torch.nn as nn

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncodingLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecodingLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model