import torch
from torch import nn, Tensor
from torch.nn import Transformer, LayerNorm
from torch.nn.functional import one_hot, dropout, log_softmax, softmax

import math
import numpy as np
from utils import DNA_TOKEN, DNA_VOCAB_SIZE, PROFILE_TOKEN, PROFILE_VOCAB_SIZE

class Positional_Encoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(Positional_Encoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
class S2S_model(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 norm_first: bool,
                 maxlen: int,
                 dropout: float = 0):
        super(S2S_model, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout, norm_first=norm_first)
        self.num_oligo = DNA_VOCAB_SIZE - 1

        self.dna_embedding = nn.Linear(self.num_oligo, emb_size,bias=False)
        self.generator = nn.Linear(emb_size, self.num_oligo)
        self.emb_size = emb_size
        

        self.src_positional_encoding = Positional_Encoding(emb_size, dropout=0, maxlen=maxlen)
        self.tgt_positional_encoding = nn.Embedding(maxlen, emb_size)

    def forward(self,
                dna: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        dna_emb = self.dna_embedding(dna)
        src_emb = self.src_positional_encoding(dna_emb)
        tgt_emb = self.tgt_positional_encoding(tgt)
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

def focal_loss_gamma(output,target,gamma):
    ce_loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
    if gamma > 0:
        pt = torch.exp(-ce_loss)
        f_loss = ((1-pt)**gamma * ce_loss)
        loss = f_loss + ce_loss
    else:
        loss = ce_loss
    return loss

def entropy(t, eps):
    k = t.reshape((-1,t.shape[-1]))
    entropy = -k * torch.log(k+eps)
    entropy = torch.sum(entropy,dim=-1)
    entropy = torch.mean(entropy)
    return entropy
