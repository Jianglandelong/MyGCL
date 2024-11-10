import torch

from GCL.losses import Loss
from GCL.losses.infonce import _similarity
from GCL.models import get_sampler

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