import torch
import torch.nn as nn

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        #print('pred', len(torch.nonzero(pred)))
        #print('target', len(torch.nonzero(target)))
        valid_mask = (target > 0).detach()
        #print('mask',len(torch.nonzero(valid_mask)))

        diff = target - pred
        #print('diff1',diff.shape)
        diff = diff[valid_mask]
        #print('diff2',diff.shape)
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
