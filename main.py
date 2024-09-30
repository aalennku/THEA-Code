import argparse
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Transformer
from model import S2S_model, focal_loss_gamma
from model import Positional_Encoding as IDS_Channel_Positional_Encoding
from train_ids_channel_dl import IDS_Channel_DL
from train_evaluate import train_epoch, evaluate
from aux import identity_aux, identity_diff_aux, identity_pos_aux, identity_diff_pos_aux, diff_aux, pos_aux

import math

from torch.utils.tensorboard import SummaryWriter

from functools import partial
import os
import sys
from timeit import default_timer as timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seq2seq encode/decode')
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--path', type=str, default='./results/')
    parser.add_argument('--path-prefix',type=str,default='')
    parser.add_argument('--path-ids-channel-dl', type=str, default='ids_channel_dl_embsize512_numlayer1_nhead4_150-200.pth')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--emb-size', type=int, default=512)
    parser.add_argument('--hid-size', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enc-layer', type=int, default=3)
    parser.add_argument('--dec-layer', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--lr-decay-step', type=int, default=600)

    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--length-dna', type=int, default=100)
    parser.add_argument('--length-codeword', type=int, default=150)
    parser.add_argument('--length-ids', help='Same to the padded_length in ids_channel_dl', 
                        type=int, default=200)
    
    parser.add_argument('--aux-q',type=float,default=1.)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--epoch',type=int,default=800)
    
    parser.add_argument('--nins', type=int, default=100, help="The proportion of insertions from all errors. ")
    parser.add_argument('--ndel', type=int, default=100)
    parser.add_argument('--nsub', type=int, default=100)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--ratio-up', type=float, default=1)
    parser.add_argument('--max-err', type=int, default=None, help="The maximum allowed number of errors in a sequence. ")
    
    parser.add_argument('--focal-loss-gamma', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm-first', type=bool, default=True)
    parser.add_argument('--aux-type',type=str, default='identity', help='Type: identity, diff, pos, identity_diff, identity_pos, identity_diff_pos')
    
    parser.add_argument('--gumbel', action="store_true", help='Use Gumbel-Softmax as constraint. ')
    parser.add_argument('--gumbel-hard',action="store_true", help='Producing hard outputs of GS. ')
    parser.add_argument('--gumbel-temperature',type=float,default=1)
    parser.add_argument('--gumbel-q',type=float,default=1)
    
    parser.add_argument('--en-resume', type=str, default=None, help="resume an encoder from this path.")
    parser.add_argument('--de-resume', type=str, default=None, help="resume an decoder from this path.")

    parser.add_argument('--train-x',type=float,default=1, help="Training stage using ratio*train_x error rate.")
    
    parser.add_argument('--ratio-mode', type=str, default=None, help="The ratio mode: None ASC DES")


    args = parser.parse_args()
    if args.aux_type == 'identity':
        args.aux_func = identity_aux
    elif args.aux_type == 'diff':
        args.aux_func = diff_aux
    elif args.aux_type == 'pos':
        args.aux_func = pos_aux
    elif args.aux_type == 'identity_diff':
        args.aux_func = identity_diff_aux
    elif args.aux_type == 'identity_pos':
        args.aux_func = identity_pos_aux
    elif args.aux_type == 'identity_diff_pos':
        args.aux_func = identity_diff_pos_aux
    else:
        print('unrecogonized aux type, using identity_aux.')
        args.aux_func = identity_aux
    
    dummy_dna = torch.zeros(args.length_dna, args.batchsize)
    dummy_aux_dna = args.aux_func(dummy_dna)
    args.length_aux = dummy_aux_dna.shape[0] # A lazy dog writes this part of code
    print('aux length: {}'.format(args.length_aux)) 

    args.maxlen = (args.length_aux + args.length_codeword + args.length_ids)
    args.device = torch.device('cuda:{}'.format(args.gpu))

    focal_loss = partial(focal_loss_gamma, gamma=args.focal_loss_gamma)
    def warmup(current_step: int):
        if current_step < args.warmup:
            return float(current_step / args.warmup)
        else:
            if args.lr_decay_step == 0:
                return 1
            else:
                return 0.1 ** np.floor((current_step - args.warmup)/args.lr_decay_step)

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    print('seed:', args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed + 10086)
    np.random.seed(args.seed + 10010)

    ids_channel_dl = torch.load(args.path_ids_channel_dl,map_location=args.device)
    ids_channel_dl.eval()

    if args.en_resume is None:
        encoder_transformer = S2S_model(args.enc_layer, args.dec_layer, args.emb_size,
                                        args.nhead, args.hid_size, 
                                        args.norm_first, args.maxlen, 
                                        dropout=args.dropout)
        for p in encoder_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        encoder_transformer = encoder_transformer.to(args.device)
    else:
        print('resume encoder.')
        encoder_transformer = torch.load(args.en_resume).to(args.device)

    if args.de_resume is None:
        decoder_transformer = S2S_model(args.enc_layer, args.dec_layer, args.emb_size,
                                        args.nhead, args.hid_size, 
                                        args.norm_first, args.maxlen, 
                                        dropout=args.dropout)
        for p in decoder_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        decoder_transformer = decoder_transformer.to(args.device)
    else:
        print('resume decoder.')
        decoder_transformer = torch.load(args.de_resume).to(args.device)
    
    loss_fn = focal_loss
    optimizer = torch.optim.Adam(list(encoder_transformer.parameters())+list(decoder_transformer.parameters()), 
                                 lr=args.lr, betas=(0.9, 0.98), eps=args.eps)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    model_path = os.path.join(args.path, str(args.seed)+'_'+args.path_prefix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    from shutil import copyfile
    copyfile(__file__,os.path.join(model_path,os.path.split(__file__)[1]))
    
    args.writer = SummaryWriter(os.path.join(model_path,'log'))
    save_args = vars(args)
    with open(os.path.join(model_path,'args.txt'),'w') as f:
        f.write(str(save_args))

    best_epoch = 0
    best_evaluate_error = np.inf
    with open(os.path.join(model_path,'logs.log'), 'w') as log_file:
        sys.stdout = log_file
        ratio = args.ratio
        for epoch in range(1, args.epoch+1):
            if (args.ratio_up!=0) and (epoch<(args.warmup*args.ratio_up)):
                ratio = ((epoch+1)/(args.warmup*args.ratio_up)) * args.ratio
            print("ratio: {}".format(ratio))
            log_file.flush()
            
            if (epoch) % 5 == 1:
                evaluate_error = evaluate(encoder_transformer, 
                                          decoder_transformer, 
                                          i=args.nins, d=args.ndel, s=args.nsub, ratio=args.ratio, args=args)
                if evaluate_error < best_evaluate_error:
                    best_evaluate_error = evaluate_error
                    best_epoch = epoch
                    torch.save(encoder_transformer, os.path.join(model_path,'encoder_best.pth'))
                    torch.save(decoder_transformer, os.path.join(model_path,'decoder_best.pth'))
                print("Evaluate_error (NER): {}, Best: {}/{}".format(evaluate_error,best_evaluate_error,best_epoch))
                args.writer.add_scalar('eval_error', evaluate_error, epoch)

            start_time = timer()
            losses = train_epoch(encoder_transformer,
                                 decoder_transformer,
                                 ids_channel_dl,
                                 optimizer,
                                 loss_fn,
                                 lr_scheduler,
                                 epoch, 
                                 ratio=ratio*args.train_x,
                                 args=args)
            end_time = timer()
            
            if (epoch) % 200 == 0:
                torch.save(encoder_transformer, os.path.join(model_path,'encoder_{}_{}.pth'.format(epoch,args.en_resume!='')))
                torch.save(decoder_transformer, os.path.join(model_path,'decoder_{}_{}.pth'.format(epoch,args.de_resume!='')))

            loss, loss_ce, loss_en, loss_aux = losses
            args.writer.add_scalar('Loss/train', loss, epoch)
            args.writer.add_scalar('Loss_ce/train', loss_ce, epoch)
            args.writer.add_scalar('Loss_en/train', loss_en, epoch)
            args.writer.add_scalar('Loss_aux/train', loss_aux, epoch)
            print((f"Epoch: {epoch}, Train loss: {loss:.4f}, loss_ce: {loss_ce:.4f}, loss_aux: {loss_aux:.4f},"f"Epoch time = {(end_time - start_time):.3f}s"))