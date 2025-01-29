import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random

from data_loader import load_data
from model import *
from utils import *

import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam

import time

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train(encoder_model, contrast_model, features, edges, optimizer, alpha, beta, gamma):
    encoder_model.train()
    optimizer.zero_grad()

    h1_lp, h1_hp, s1_lp, s1_hp, z2_lp, z2_hp = encoder_model(features, edges)
    lp_loss = contrast_model(h_pred = h1_lp, h_target = z2_lp.detach())
    hp_loss = contrast_model(h_pred = h1_hp, h_target = z2_hp.detach())
    cross_view_loss = lp_loss * alpha + hp_loss * beta
    cross_pass_loss = (contrast_model(h_pred = s1_lp, h_target = z2_hp) + contrast_model(h_pred = s1_hp, h_target = z2_lp)) * (gamma/2)
    loss = cross_view_loss + cross_pass_loss
    
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.0)
    return loss.item()


def main(args):
    start_time = time.time()
    device = torch.device('cuda')

    setup_seed(0)
    features, edges, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(args.dataset)
    results = []
    results_2 = []

    aug1 = A.Compose([A.EdgeRemoving(pe=args.dropedge_rate_1), A.FeatureMasking(pf=args.maskfeat_rate_1), A.NodeDropping(pn=args.dropnode_rate_1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=args.dropedge_rate_2), A.FeatureMasking(pf=args.maskfeat_rate_2), A.NodeDropping(pn=args.dropnode_rate_2)])

    for trial in range(args.ntrials):
        setup_seed(trial)

        # gconv = GConv(input_dim=nfeats, hidden_dim=256, num_layers=2).to(device)
        gconv = GConv(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim, proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor1=aug1, augmentor2=aug2, nnodes=nnodes, hidden_dim=args.proj_dim, sparse=args.sparse, dropout=args.dropout).to(device)
        contrast_model = CrossViewContrast().to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.001)

        features = features.to(device)
        # str_encodings = str_encodings.to(device)
        edges = edges.to(device)

        best_acc_val = 0
        best_acc_test = 0
        best_acc_test_2 = 0

        for epoch in range(1, args.epochs + 1):
            loss = train(encoder_model, contrast_model, features, edges, optimizer, args.alpha, args.beta, args.gamma)
            # print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} ".format(epoch, loss))

            if epoch % args.eval_freq == 0:
                encoder_model.eval()
                # h_lp, h_hp, _, _, _, _ = encoder_model(features, edges)
                emb_lp, emb_hp = encoder_model.get_embedding(features, edges)
                z = torch.cat([emb_lp * args.alpha, emb_hp * args.beta], dim=1)
                
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(z, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                if args.full_log == 1:
                    print('[TEST] Trial:{:04d} | Epoch:{:04d} | CL loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                            trial, epoch, loss, acc_val, acc_test))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test
                if acc_test > best_acc_test_2:
                    best_acc_test_2 = acc_test

        results.append(best_acc_test)
        results_2.append(best_acc_test_2)

    for i, acc in enumerate(results):
        print('Trial:{} | ACC:{:.2f} | ACC2:{:.2f}'.format(i, acc, results_2[i]))
    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f} | ACC2:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results), np.std(results), np.mean(results_2), np.std(results_2)))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='cornell',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
                                 'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics'])
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=20)
    parser.add_argument('-epochs', type=int, default=400)
    # parser.add_argument('-lr_gcl', type=float, default=0.001)
    # parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-lr', type=float, default=0.001)
    # parser.add_argument('-cl_rounds', type=int, default=2)
    # parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)
    # parser.add_argument('-subgraph', type=bool, default=False)

    # DISC Module - Hyper-param
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-gamma', type=float, default=0.5)
    # parser.add_argument('-margin_hom', type=float, default=0.5)
    # parser.add_argument('-margin_het', type=float, default=0.5)

    # GRL Module - Hyper-param
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    # parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)
    parser.add_argument('-dropnode_rate_1', type=float, default=0.1)
    parser.add_argument('-dropnode_rate_2', type=float, default=0.1)
    parser.add_argument('-full_log', type=int, default=1)

    args = parser.parse_args()

    print(args)
    main(args)