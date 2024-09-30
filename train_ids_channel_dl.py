from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.functional import one_hot, softmax, log_softmax
from scipy.special import softmax as np_softmax
import numpy as np
from utils import *
from timeit import default_timer as timer
from model import Positional_Encoding as IDS_Channel_Positional_Encoding

class IDS_Channel_DL(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(IDS_Channel_DL, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, DNA_VOCAB_SIZE)
        self.emb_size = emb_size
        self.dna_embedding = nn.Linear(DNA_VOCAB_SIZE,emb_size//2,bias=False)
        self.profile_embedding = nn.Linear(PROFILE_VOCAB_SIZE,emb_size//2,bias=False)
        self.tgt_embedding = nn.Linear(DNA_VOCAB_SIZE,emb_size,bias=False)
        self.positional_encoding = IDS_Channel_Positional_Encoding(
            emb_size, dropout=dropout)

    def forward(self,
                dna: Tensor,
                profile: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):

        dna_emb = self.dna_embedding(dna)
        profile_emb = self.profile_embedding(profile)
        src_emb = torch.cat([dna_emb,profile_emb],dim=-1)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(torch.zeros(size=(*tgt.shape[:-1], self.emb_size),device=src_emb.device))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = torch.zeros((tgt_seq_len, tgt_seq_len),device=DEVICE).type(torch.bool)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = torch.zeros(size=src.shape[:2],device=DEVICE).transpose(0, 1)
    tgt_padding_mask = torch.zeros(size=tgt.shape[:2],device=DEVICE).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def generate_sample(length, padded_length, i, d, s, ratio):
    subs = s
    s = np.random.randint(0,4,size=length).astype(int)
    s = np_one_hot(s,4)
    s = s*5 + (np.random.rand(*s.shape)-0.5)*10
    s = np_softmax(s,axis=-1)
    s = s.tolist()
    profile = random_profile(length, i, d, subs, ratio)
    target = ids_channel_OH(s, profile)
    if len(target) > padded_length:
        target = target[:padded_length]
    if len(profile) > padded_length:
        profile = profile[:padded_length]
    
    s = [list(_)+[0] for _ in s]
    target = [list(_)+[0] for _ in target]
    for _ in range(padded_length-len(s)):
        s.append([0,0,0,0,1])
    for _ in range(padded_length-len(profile)):
        profile.append('<PAD>')
    for _ in range(padded_length-len(target)):
        target.append([0,0,0,0,1])
    return s, profile, target
    
def generate_sample_batch(batch_size, length, padded_length, i, d, s, ratio):
    dna_b = []
    profile_b = []
    target_b = []
    for _ in range(batch_size):
        dna, profile, target = generate_sample(length, padded_length, i=i, d=d, s=s, ratio=ratio)
        profile = [PROFILE_TOKEN[_] for _ in profile]
        dna_b.append(dna)
        profile_b.append(profile)
        target_b.append(target)
    return torch.tensor(dna_b).transpose(0,1),torch.tensor(profile_b).transpose(0,1),torch.tensor(target_b).transpose(0,1)

def warmup(current_step: int):
    if current_step < WARMUP_STEPS:
        return float(current_step / WARMUP_STEPS)
    else:
        return 1
    
def train_epoch(model, optimizer, loss_fn):
    model.train()
    losses = 0
    N = 256
    for _ in range(N):
        dna,profile,tgt = generate_sample_batch(batch_size=BATCH_SIZE,
                                                length=LENGTH,
                                                padded_length=PADDED_LENGTH,
                                                i=10, d=10, s=10, ratio=0.01)
        dna = dna.to(DEVICE)
        profile = profile.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(dna[:,:,0], tgt[:,:,0])
        profile = one_hot(profile,num_classes=PROFILE_VOCAB_SIZE).float()        
        logits = model(dna, profile, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt
        log_prob = log_softmax(logits, dim=-1)
        loss = loss_fn(log_prob.reshape(-1, log_prob.shape[-1]), tgt_out.reshape(-1, tgt_out.shape[-1]))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / N

if __name__ == '__main__':
    WARMUP_STEPS = 400
    NHEAD = 4
    EMB_SIZE = 512
    FFN_HID_DIM = 512
    BATCH_SIZE = 32
    NUM_ENCODER_LAYERS = 1
    NUM_DECODER_LAYERS = 1
    LENGTH = 150
    PADDED_LENGTH = 200
    DEVICE = torch.device('cuda:1')
    DROPOUT = 0
    NUM_EPOCHS = 1

    ids_channel_dl = IDS_Channel_DL(NUM_ENCODER_LAYERS, 
                                    NUM_DECODER_LAYERS, 
                                    EMB_SIZE,
                                    NHEAD, 
                                    FFN_HID_DIM,
                                    dropout=DROPOUT)

    for p in ids_channel_dl.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    ids_channel_dl = ids_channel_dl.to(DEVICE)
    loss_fn = torch.nn.KLDivLoss()
    optimizer = torch.optim.Adam(ids_channel_dl.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(1, NUM_EPOCHS+1):
        if epoch % 400 == 0:
            optimizer.param_groups[0]['lr'] *= 0.1
        start_time = timer()
        train_loss = train_epoch(ids_channel_dl, optimizer, loss_fn)
        end_time = timer()
        print((f"Epoch: {epoch}, Train loss: {train_loss:.10f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    torch.save(ids_channel_dl,
               'ids_channel_dl_embsize{}_numlayer{}_nhead{}_{}-{}.pth'.format(EMB_SIZE, 
                                                                              NUM_DECODER_LAYERS, 
                                                                              NHEAD,
                                                                              LENGTH,
                                                                              PADDED_LENGTH))