import torch
from torch.nn.functional import one_hot, dropout
from torch import nn, Tensor
from torch.nn import LayerNorm, Transformer
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.functional import log_softmax, softmax

from model import entropy

import numpy as np

from utils import DNA_VOCAB_SIZE, PROFILE_VOCAB_SIZE, ids_channel

from dataset import Random_DNA_dataset, generate_profile_batch

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = torch.zeros((tgt_seq_len, tgt_seq_len),device=device).type(torch.bool)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
    src_padding_mask = torch.zeros(size=src.shape[:2],device=device).transpose(0, 1)
    tgt_padding_mask = torch.zeros(size=tgt.shape[:2],device=device).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def translate(model: torch.nn.Module, dna, profile, args):
    model.eval()
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(dna, dna, args.device)
    if len(dna.shape) == 2:
        dna = one_hot(dna.long(),num_classes=DNA_VOCAB_SIZE).float()
    profile = one_hot(profile.long(),num_classes=PROFILE_VOCAB_SIZE).float()
    dna = dna.to(args.device)
    profile = profile.to(args.device)
    logits = model(dna, profile, dna, 
                   src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
    return logits

def get_temperature(inputs, args):
#     print(inputs.shape)
    if args.temperature_flag == 'random_nucleotide':
        t = torch.rand(inputs.size()[:-1])
        t = t*(args.tmax-args.tmin) + args.tmin
    if args.temperature_flag == 'random_strand':
        t = torch.rand(inputs.size()[1:-1])
        t = t*(args.tmax-args.tmin) + args.tmin
    if args.temperature_flag == 'sin':
        t = 1
    return t.to(args.device)

def train_epoch(encoder, decoder, ids_channel_dl, 
                optimizer, loss_fn, lr_scheduler, 
                epoch, ratio,
                args, decoder_2=None, IDX_partial_codeword=None):
    print('learning rate: {}'.format(lr_scheduler.get_last_lr()[0]))
    print('ratio: {} {}'.format(ratio,args.ratio_mode))
    encoder.train()
    decoder.train()
    ids_channel_dl.eval()
    num_oligo = DNA_VOCAB_SIZE - 1
    losses = 0
    losses_ce = 0
    losses_en = 0
    losses_aux = 0
    
    gumbel_temperature = args.gumbel_temperature
    print('gumbel_temperature', gumbel_temperature)
    
    N = 256
    SHOW = 0

    dna_dataset = Random_DNA_dataset(N=N*args.batchsize, length_dna=args.length_dna)
    dna_dataset_loader = torch.utils.data.DataLoader(dna_dataset, batch_size=args.batchsize, num_workers=20)

    for _, dna in enumerate(dna_dataset_loader):
        dna = dna.transpose(0,1)
        dna = dna.to(args.device)

        position = torch.arange(0,args.length_aux+args.length_codeword).long().to(args.device).unsqueeze(-1).expand(args.length_aux+args.length_codeword,args.batchsize)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(dna, position, args.device)
        dna_oh = one_hot(dna,num_classes=num_oligo).float()
        tgt = dna_oh

        aux_tgt = args.aux_func(dna)
        aux_tgt = one_hot(aux_tgt,num_classes=num_oligo).float()

        encoder_logits = encoder(dna_oh, position, src_mask, tgt_mask, 
                                 src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        auxword_logits = encoder_logits[:args.length_aux,:,:]
        codeword_logits = encoder_logits[args.length_aux:,:,:]
        if _ == 0: 
            print('encoder_logits')
            print(encoder_logits[args.length_aux,0,:].cpu().detach().numpy())
            print('DNA shape')
            print(dna.shape)
            print('encoder_logits_softmax')
            encoder_logits_softmax = softmax(encoder_logits,dim=-1)
            print(encoder_logits_softmax[args.length_aux,0,:].cpu().detach().numpy())
            
        if args.gumbel:
            if args.gumbel_q == 1:
                codeword = F.gumbel_softmax(codeword_logits, tau=gumbel_temperature, hard=args.gumbel_hard)
            elif args.gumbel_q == 0:
                codeword = softmax(codeword_logits,dim=-1)
            else:
                codeword_GS = F.gumbel_softmax(codeword_logits, tau=gumbel_temperature, hard=args.gumbel_hard)
                codeword_softmax = softmax(codeword_logits,dim=-1)
                codeword = args.gumbel_q*codeword_GS + (1-args.gumbel_q)*codeword_softmax

        else:
            codeword = softmax(codeword_logits,dim=-1)

        # compute aux loss
        loss_aux = torch.nn.functional.cross_entropy(auxword_logits.reshape(-1, auxword_logits.shape[-1]), 
                                                     aux_tgt.reshape(-1, aux_tgt.shape[-1]))
        losses_aux += loss_aux.item()
        # aux loss end
        
        # compute entropy
        codeword_softmax = softmax(codeword_logits,dim=-1)
        loss_en = entropy(codeword_softmax,args.eps)
        losses_en += loss_en.item()
        # entropy end

        codeword_before_ids = torch.cat([codeword, 
                                         torch.zeros(size=[*codeword.shape[:2],1]).to(args.device)],
                                         dim=-1)
        pad = one_hot(torch.tensor([[num_oligo]]),num_classes=DNA_VOCAB_SIZE).to(args.device)
        codeword_before_ids = torch.cat([codeword_before_ids, 
                                         pad.expand(args.length_ids-args.length_codeword,args.batchsize,-1)],
                                         dim=0)
        if _ == 0:
            print('codeword_before_ids')
            print(codeword_before_ids[SHOW,0,:].cpu().detach().numpy())

        profile = generate_profile_batch(batch_size=args.batchsize,
                                         length=args.length_codeword,
                                         padded_length=args.length_ids,
                                         i=args.nins, d=args.ndel, s=args.nsub,
                                         ratio=ratio, ratio_mode=args.ratio_mode,
                                         max_err=args.max_err).to(args.device)
        
        if _ == 0:
            print('profile errors')
            p = profile.detach().cpu().numpy()
            print(np.sum(p<4)-np.sum(p==0),np.sum(p<8)-np.sum(p<4),np.sum(p==8))
        
        ids_logits = translate(ids_channel_dl, codeword_before_ids, profile, args)
        ids_codeword = softmax(ids_logits,dim=-1)
        ids_codeword = ids_codeword[:,:,:4]
        
        if _ == 0:
            print('ids_codeword')
            print(ids_codeword[SHOW,0,:].cpu().detach().numpy())
        
        decode_position = torch.arange(0,args.length_dna).long().to(args.device).unsqueeze(-1).expand(args.length_dna,args.batchsize)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(ids_codeword[:,:,0], decode_position, args.device)
        
        decoder_logits = decoder(ids_codeword, decode_position, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        if _ == 0:
            print('decoder_logits')
            print(decoder_logits[SHOW,0,:].cpu().detach().numpy())
            
        #cross entropy or focal loss on reconstructed word
        loss_ce = loss_fn(decoder_logits.reshape(-1, decoder_logits.shape[-1]), 
                        tgt.reshape(-1, tgt.shape[-1]))
        loss_ce_mean = loss_ce.mean()
        losses_ce += loss_ce_mean.item()
        
        loss_sum = loss_ce_mean
        loss_sum += args.aux_q * loss_aux
            
        optimizer.zero_grad()
        loss_sum.backward()
        losses += loss_sum.item()
        optimizer.step()
    lr_scheduler.step()

    return losses/N, losses_ce/N, losses_en/N, losses_aux/N

def ner(encoder,decoder,i,d,s,ratio,args):
    dna, pred, _0, _1 = test(encoder,decoder,i,d,s,ratio,args)
    diff = (pred - dna).detach().cpu().numpy()
    return np.sum(diff!=0)/len(diff.flatten())

def test(encoder,decoder,i,d,s,ratio,args):
    encoder.eval()
    decoder.eval()
    num_oligo = DNA_VOCAB_SIZE - 1
    dna = torch.tensor(np.random.randint(0,4,size=(args.length_dna, args.batchsize))).to(args.device)

    position = torch.arange(0,args.length_aux+args.length_codeword).long().to(args.device).unsqueeze(-1).expand(args.length_aux+args.length_codeword,args.batchsize)
    
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(dna, position, args.device)
    dna_oh = one_hot(dna,num_classes=num_oligo).float()
    logits = encoder(dna_oh, position, 
                     src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    codeword = softmax(logits,dim=-1)[args.length_aux:]
    a,b = torch.max(codeword,dim=-1)
    # b is the codeword in letters
    
    profile = generate_profile_batch(batch_size=args.batchsize,
                                     length=args.length_codeword,
                                     padded_length=args.length_ids,
                                     i=i,d=d,s=s,ratio=ratio,ratio_mode=args.ratio_mode,max_err=args.max_err)
    b_np = b.cpu().detach().numpy()
    
    b_ids = []
    for a, p in zip(b_np.T,profile.T):
        bb = ids_channel(a,p)
        if len(bb) < args.length_ids:
            bb = bb + [4]*(args.length_ids-len(bb))
        else:
            bb = bb[:args.length_ids]
        b_ids.append(bb)
#     new_b is the codeword in letters after IDS according to profile p
        
    b_ids = torch.tensor(b_ids).to(args.device).T
    b_ids_oh = one_hot(b_ids,num_classes=5).to(torch.float)
    b_ids_oh = b_ids_oh[:,:,:4]
    b_ids_oh = b_ids_oh
    decode_position = torch.arange(0,args.length_dna).long().to(args.device).unsqueeze(-1).expand(args.length_dna,args.batchsize)
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(b_ids_oh[:,:,0], decode_position, args.device)

    logits = decoder(b_ids_oh, decode_position, src_mask, tgt_mask, 
                     src_padding_mask, tgt_padding_mask, src_padding_mask)
    _, pred = torch.max(logits,dim=-1)
    
    return dna, pred, b, b_ids


def evaluate(encoder,decoder,i,d,s,ratio,args):
    encoder.eval()
    decoder.eval()
    ner_list = []
    for _ in range(50):
        ner_list.append(ner(encoder,decoder,i=i,d=d,s=s,ratio=ratio,args=args))
    return(np.mean(ner_list))

