import argparse
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Transformer
from model import S2S_model, focal_loss_gamma
from model import Positional_Encoding as IDS_Channel_Positional_Encoding
from train_ids_channel_dl import IDS_Channel_DL
from train_evaluate import train_epoch, evaluate, test


import math

from torch.utils.tensorboard import SummaryWriter

from functools import partial
import os
import sys
from timeit import default_timer as timer
from model import S2S_model
from tqdm import tqdm

def to_string(a):
    a = [str(_) for _ in a]
    return ''.join(a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seq2seq encode/decode')
    parser.add_argument('--gpu', type=int, default='1')
    parser.add_argument('--model-path', type=str, default='.')
    parser.add_argument('--length-dna', type=int, default=100)
    parser.add_argument('--length-codeword', type=int, default=150)
    parser.add_argument('--length-ids', help='same to the padded_length in ids_channel_dl', 
                        type=int, default=200)
    parser.add_argument('--nins', type=int, default=10, help="The proportion of insertions from all errors. ")
    parser.add_argument('--ndel', type=int, default=10)
    parser.add_argument('--nsub', type=int, default=10)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--ratio-mode', type=str, default=None, help="The ratio mode: None ASC DES")
    parser.add_argument('--max-err', type=int, default=None, help="The maximum allowed number of errors in a sequence. ")

    args = parser.parse_args()
    args.length_aux = args.length_dna
    args.device = torch.device('cuda:{}'.format(args.gpu))

    encoder = torch.load(os.path.join(args.model_path,'encoder_best.pth')).to(args.device)
    decoder = torch.load(os.path.join(args.model_path,'decoder_best.pth')).to(args.device)

    ner_list = []
    for _ in tqdm(range(800)):
        dna, pred, codeword, codeword_ids = test(encoder,decoder,
                         args.nins,args.ndel,args.nsub,args.ratio,args)
        dna = dna.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        codeword = codeword.cpu().detach().numpy()
        codeword_ids = codeword_ids.cpu().detach().numpy()
        diff = dna - pred
        ner_list.append(np.sum(diff!=0)/len(diff.flatten()))
        cnt = 0
        with open(os.path.join(args.model_path,'test_results.csv'),'a') as f:
            for a,b,c,d in zip(dna.T, pred.T, codeword.T, codeword_ids.T):
                if cnt == 1:
                    break
                cnt += 1
                a = to_string(a)
                b = to_string(b)
                c = to_string(c)
                d = to_string(d)
                f.write(f'{a},{b},{c},{d}\n')

    np.save(os.path.join(args.model_path,'{}_NER_{}.npy'.format(args.ratio_mode, np.mean(ner_list))), ner_list)
    print('NER: {}'.format(np.mean(ner_list)))


