import torch
from torch import Tensor

def identity_aux(dna):
    return dna

def diff_aux(dna):
    to_cat = []
    for _ in [1,]:
        dna_1 = (dna+4-dna.roll(_,dims=0))%4
        to_cat.append(dna_1.clone())
    return torch.concatenate(to_cat, dim=0)

def pos_aux(dna):
    return (dna + torch.arange(0,dna.shape[0]).unsqueeze(-1).expand(*dna.shape).to(dna.device))%4

def identity_diff_aux(dna):
    to_cat = [dna,]
    for _ in [1,]:
        dna_1 = (dna+4-dna.roll(_,dims=0))%4
        to_cat.append(dna_1.clone())
    return torch.concatenate(to_cat, dim=0)

def identity_pos_aux(dna):
    pos = (dna + torch.arange(0,dna.shape[0]).unsqueeze(-1).expand(*dna.shape).to(dna.device))%4
    return torch.concatenate([dna,pos],dim=0)

def identity_diff_pos_aux(dna):
    idt_diff = identity_diff_aux(dna)
    pos = (dna + torch.arange(0,dna.shape[0]).unsqueeze(-1).expand(*dna.shape).to(dna.device))%4
    return torch.concatenate([idt_diff,pos],dim=0)
