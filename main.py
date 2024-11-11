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

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train(encoder_model, contrast_model, features, edges, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(features, edges)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


# def train_cl(cl_model, discriminator, optimizer_cl, features, str_encodings, edges):

#     cl_model.train()
#     discriminator.eval()

#     adj_1, adj_2, weights_lp, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
#     features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, args, cl_model.training)
#     cl_loss = cl_model(features_1, adj_1, features_2, adj_2)

#     optimizer_cl.zero_grad()
#     cl_loss.backward()
#     optimizer_cl.step()

#     return cl_loss.item()


# def main():
#     device = torch.device('cuda')
#     path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
#     dataset = WikiCS(path, transform=T.NormalizeFeatures())
#     data = dataset[0].to(device)

#     aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
#     aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

#     gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, num_layers=2).to(device)
#     encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
#     contrast_model = MultiViewContrast(loss=L.InfoNCE(tau=0.2), intra_loss=IntraInfoNCE(tau=0.2), beta=0.2, mode='L2L').to(device)

#     optimizer = Adam(encoder_model.parameters(), lr=0.01)

#     with tqdm(total=100, desc='(T)') as pbar:
#         for epoch in range(1, 101):
#             loss = train(encoder_model, contrast_model, data, optimizer)
#             pbar.set_postfix({'loss': loss})
#             pbar.update()

#     test_result = test(encoder_model, data)
#     print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


def main(args):
    device = torch.device('cuda')

    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats = load_data(args.dataset)
    results = []

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    for trial in range(args.ntrials):
        setup_seed(trial)

        gconv = GConv(input_dim=nfeats, hidden_dim=256, num_layers=2).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor1=aug1, augmentor2=aug2, hidden_dim=256).to(device)
        contrast_model = MultiViewContrast(loss=L.InfoNCE(tau=0.2), intra_loss=IntraInfoNCE(tau=0.2), beta=0.2, mode='L2L').to(device)

        optimizer = Adam(encoder_model.parameters(), lr=0.01)

        # cl_model = GCL(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
        #             proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).to(device)
        # optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)

        features = features.to(device)
        # str_encodings = str_encodings.to(device)
        edges = edges.to(device)

        best_acc_val = 0
        best_acc_test = 0

        for epoch in range(1, args.epochs + 1):
            loss = train(encoder_model, contrast_model, features, edges, optimizer)
            print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} ".format(epoch, loss))

            if epoch % args.eval_freq == 0:
                encoder_model.eval()
                h1, h2, _, _, _, _ = encoder_model(features, edges)
                z = torch.cat([h1, h2], dim=1)
                
                # adj_1, adj_2, _, _ = discriminator(torch.cat((features, str_encodings), 1), edges)
                # embedding = cl_model.get_embedding(features, adj_1, adj_2)
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(z, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                print('[TEST] Epoch:{:04d} | CL loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                        epoch, loss, acc_val, acc_test))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test

        results.append(best_acc_test)

    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results),
                                                                           np.std(results)))

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
    parser.add_argument('-lr_gcl', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)

    # DISC Module - Hyper-param
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)

    # GRL Module - Hyper-param
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)

    args = parser.parse_args()

    print(args)
    main(args)