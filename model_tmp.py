import torch
import copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU

import GCL.losses as L
from GCL.losses import Loss
from GCL.losses.infonce import _similarity
from GCL.models import get_sampler
from utils import *
from dgl.nn import EdgeWeightNorm

import torch.nn as nn
import dgl.function as fn


EOS = 1e-10
norm = EdgeWeightNorm(norm='both')

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


# 单层GCN后接batch norm，加预测头
class GConv(torch.nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse, batch_size):
        super(GConv, self).__init__()
        self.encoder = SGC(nlayers, in_dim, emb_dim, dropout, sparse)

        # if nlayers_proj == 1:
        #     self.proj_head = Sequential(Linear(emb_dim, proj_dim))
        # elif nlayers_proj == 2:
        #     self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
        layers = []
        for i in range(nlayers_proj):
            layers.append(Linear(emb_dim if len(layers) == 0 else proj_dim, proj_dim))
            if i < nlayers_proj - 1:
                layers.append(ReLU(inplace=True))
        self.proj_head = Sequential(*layers)

        self.batch_norm = Normalize(emb_dim, norm='batch')
        self.batch_size = batch_size
        
    def forward(self, x, a, edge_weight=None):
        z = x
        z = self.encoder(z, a)
        z = self.batch_norm(z)
        return z, self.proj_head(z)
        
    # def forward(self, x, edge_index, edge_weight=None):
    #     z = x
    #     for conv in self.layers:
    #         z = conv(z, edge_index, edge_weight)
    #         z = self.activation(z)
    #         z = F.dropout(z, p=self.dropout, training=self.training)
    #     z = self.batch_norm(z)
    #     return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor1, augmentor2, nnodes, hidden_dim, sparse, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2
        self.predictor1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.predictor2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))
        self.nnodes = nnodes
        self.sparse = sparse

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p
    
    def get_adj(self, edges):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, self.nnodes)
            adj_lp += torch.eye(self.nnodes).cuda()
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, self.nnodes)
            adj_hp += torch.eye(self.nnodes).cuda()
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            # Note: 这里的mask是GREET中设置掩膜参数的，可调，如果不需要掩膜，可以直接减adj_hp
            mask = torch.zeros(adj_lp.shape).cuda()
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes).cuda() - adj_hp * mask
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((torch.ones(edges.shape[1]).cuda(), torch.ones(self.nnodes).cuda())) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp

            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((torch.ones(edges.shape[1]).cuda(), torch.ones(self.nnodes).cuda())) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= -1
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp

    def forward(self, x, edge_index, edge_weight=None):
        x1, edge_index1, edge_weight1 = self.augmentor1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.augmentor2(x, edge_index, edge_weight)
        adj1_lp, adj1_hp = self.get_adj(edge_index1)
        adj2_lp, adj2_hp = self.get_adj(edge_index2)

        _, z1_lp = self.online_encoder(x1, adj1_lp, edge_weight1)
        _, z1_hp = self.online_encoder(x1, adj1_hp, edge_weight1)
        
        h1_lp = self.predictor1(z1_lp)
        h1_hp = self.predictor1(z1_hp)

        s1_lp = self.predictor2(z1_lp)
        s1_hp = self.predictor2(z1_hp)

        with torch.no_grad():
            _, z2_lp = self.get_target_encoder()(x2, adj2_lp, edge_weight2)
            _, z2_hp = self.get_target_encoder()(x2, adj2_hp, edge_weight2)
        
        return h1_lp, h1_hp, s1_lp, s1_hp, z2_lp, z2_hp

    def get_embedding(self, x, edge_index, edge_weight=None):
        adj_lp, adj_hp = self.get_adj(edge_index)
        emb_lp, _ = self.online_encoder(x, adj_lp, edge_weight)
        emb_hp, _ = self.online_encoder(x, adj_hp, edge_weight)
        return emb_lp, emb_hp



# class MultiViewContrast(torch.nn.Module):
#     def __init__(self, loss, intra_loss, beta, mode='L2L'):
#         super(MultiViewContrast, self).__init__()
#         self.loss = loss
#         self.intra_loss = intra_loss
#         self.beta = beta
#         self.mode = mode
#         self.sampler = get_sampler(mode, intraview_negs=False)

#     def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
#                 g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
#                 batch=None, extra_pos_mask=None):
#         assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
#         anchor1, sample1, pos_mask, neg_mask = self.sampler(anchor=h1_pred, sample=h2_target)
#         anchor2, sample2 = h2_pred, h1_target

#         loss_cn_1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask, neg_mask=neg_mask)
#         loss_cn_2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask, neg_mask=neg_mask)
#         cross_network_loss = (loss_cn_1 + loss_cn_2) * 0.5

#         inter_anchor1, inter_sample1 = h1_pred, h2_pred
#         inter_anchor2, inter_sample2 = h2_pred, h1_pred
#         loss_inter_1 = self.loss(anchor=inter_anchor1, sample=inter_sample1, pos_mask=pos_mask, neg_mask=neg_mask)
#         loss_inter_2 = self.loss(anchor=inter_anchor2, sample=inter_sample2, pos_mask=pos_mask, neg_mask=neg_mask)

#         intra_anchor1, intra_sample1 = h1_pred, h2_pred
#         intra_anchor2, intra_sample2 = h1_pred, h2_pred
#         loss_intra_1 = self.intra_loss(anchor=intra_anchor1, sample=intra_sample1, pos_mask=pos_mask, neg_mask=neg_mask)
#         loss_intra_2 = self.intra_loss(anchor=intra_anchor2, sample=intra_sample2, pos_mask=pos_mask, neg_mask=neg_mask)

#         cross_view_loss = (loss_inter_1 + loss_inter_2 + loss_intra_1 + loss_intra_2) * 0.5

#         return self.beta * cross_view_loss + (1 - self.beta) * cross_network_loss
    
# class IntraInfoNCE(Loss):
#     def __init__(self, tau):
#         super(IntraInfoNCE, self).__init__()
#         self.tau = tau

#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
#         self_view_sim = _similarity(anchor, anchor) / self.tau
#         cross_view_sim = _similarity(anchor, sample) / self.tau
#         exp_sim = torch.exp(cross_view_sim) * pos_mask + torch.exp(self_view_sim) * neg_mask
#         log_prob = cross_view_sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = log_prob * pos_mask
#         loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
#         return -loss.mean()
    


class CrossViewContrast(torch.nn.Module):
    def __init__(self):
        super(CrossViewContrast, self).__init__()
        self.loss = L.BootstrapLatent()
        self.sampler = get_sampler('L2L', intraview_negs=False)

    def forward(self, h_pred, h_target):
        anchor, sample, pos_mask, _ = self.sampler(anchor=h_pred, sample=h_target)
        l = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask)
        return l



# 单层GCN，不带激活函数，先对x激活
class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse

        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    def forward(self, x, g):
        x = torch.relu(self.linear(x))

        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x