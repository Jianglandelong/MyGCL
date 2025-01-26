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

        if nlayers_proj == 1:
            self.proj_head = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

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
        self.encoder = encoder
        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2
        self.nnodes = nnodes
        self.sparse = sparse

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

        h1_lp, h1_lp_pred = self.encoder(x1, adj1_lp, edge_weight1)
        h1_hp, h1_hp_pred = self.encoder(x1, adj1_hp, edge_weight1)
        
        # h1_lp_pred = self.predictor(h1_lp_online)
        # h1_hp_pred = self.predictor(h1_hp_online)

        _, h2_lp_pred = self.encoder(x2, adj2_lp, edge_weight2)
        _, h2_hp_pred = self.encoder(x2, adj2_hp, edge_weight2)
        
        return h1_lp, h1_hp, h1_lp_pred, h1_hp_pred, h2_lp_pred, h2_hp_pred
    
    def get_embedding(self, x, edge_index, edge_weight=None):
        adj_lp, adj_hp = self.get_adj(edge_index)
        h_lp, _ = self.encoder(x, adj_lp, edge_weight)
        h_hp, _ = self.encoder(x, adj_hp, edge_weight)
        return h_lp, h_hp


class CrossViewContrast(torch.nn.Module):
    def __init__(self):
        super(CrossViewContrast, self).__init__()
        self.loss = L.BootstrapLatent()
        self.sampler = get_sampler('L2L', intraview_negs=False)

    def forward(self, h_pred, h_target):
        anchor, sample, pos_mask, _ = self.sampler(anchor=h_pred, sample=h_target)
        l = self.loss(anchor=anchor, sample=sample, pos_mask=pos_mask)
        return l
    
    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss


    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()



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