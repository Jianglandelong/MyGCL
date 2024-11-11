import torch
import copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from GCL.losses import Loss
from GCL.losses.infonce import _similarity
from GCL.models import get_sampler


# import os.path as osp
# import GCL.losses as L
# import GCL.augmentors as A
# import torch_geometric.transforms as T

# from tqdm import tqdm
# from torch.optim import Adam
# from GCL.eval import get_split, LREvaluator
# from GCL.models import BootstrapContrast
# from torch_geometric.datasets import WikiCS


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


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor1, augmentor2, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

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

    def forward(self, x, edge_index, edge_weight=None):
        x1, edge_index1, edge_weight1 = self.augmentor1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.augmentor2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target



class MultiViewContrast(torch.nn.Module):
    def __init__(self, loss, intra_loss, beta, mode='L2L'):
        super(MultiViewContrast, self).__init__()
        self.loss = loss
        self.intra_loss = intra_loss
        self.beta = beta
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=False)

    def forward(self, h1_pred=None, h2_pred=None, h1_target=None, h2_target=None,
                g1_pred=None, g2_pred=None, g1_target=None, g2_target=None,
                batch=None, extra_pos_mask=None):
        if self.mode == 'L2L':
            assert all(v is not None for v in [h1_pred, h2_pred, h1_target, h2_target])
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1_pred, sample=h2_target)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2_pred, sample=h1_target)

        loss_cn_1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1)
        loss_cn_2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2)
        cross_network_loss = (loss_cn_1 + loss_cn_2) * 0.5

        inter_anchor1, inter_sample1, inter_pos_mask1, inter_neg_mask1 = self.sampler(anchor=h1_pred, sample=h2_pred)
        inter_anchor2, inter_sample2, inter_pos_mask2, inter_neg_mask2 = self.sampler(anchor=h2_pred, sample=h1_pred)
        loss_inter_1 = self.loss(anchor=inter_anchor1, sample=inter_sample1, pos_mask=inter_pos_mask1, neg_mask=inter_neg_mask1)
        loss_inter_2 = self.loss(anchor=inter_anchor2, sample=inter_sample2, pos_mask=inter_pos_mask2, neg_mask=inter_neg_mask2)

        intra_anchor1, intra_sample1, intra_pos_mask1, intra_neg_mask1 = self.sampler(anchor=h1_pred, sample=h2_pred)
        intra_anchor2, intra_sample2, intra_pos_mask2, intra_neg_mask2 = self.sampler(anchor=h1_pred, sample=h2_pred)
        loss_intra_1 = self.intra_loss(anchor=intra_anchor1, sample=intra_sample1, pos_mask=intra_pos_mask1, neg_mask=intra_neg_mask1)
        loss_intra_2 = self.intra_loss(anchor=intra_anchor2, sample=intra_sample2, pos_mask=intra_pos_mask2, neg_mask=intra_neg_mask2)

        cross_view_loss = (loss_inter_1 + loss_inter_2 + loss_intra_1 + loss_intra_2) * 0.5

        return self.beta * cross_view_loss + (1 - self.beta) * cross_network_loss
    
class IntraInfoNCE(Loss):
    def __init__(self, tau):
        super(IntraInfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        self_view_sim = _similarity(anchor, anchor) / self.tau
        cross_view_sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(cross_view_sim) * pos_mask + torch.exp(self_view_sim) * neg_mask
        log_prob = cross_view_sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()