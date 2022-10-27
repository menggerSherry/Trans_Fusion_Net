from turtle import forward
from cv2 import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import utils

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss','CE_Loss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class BoundaryMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input,edge):
        bce = F.binary_cross_entropy_with_logits(input, edge)
        return bce 

class BoundaryRegular(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    def forward(self,seg,edge):
        edge = (edge>0.7).float().detach()
        seg = seg*edge
        return self.l1_loss(seg,edge)

class SegmentationRegular(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,seg,edge):
        seg_label = ((seg+edge)>0.7).float().detach()
        bce = F.binary_cross_entropy_with_logits(seg,seg_label)
        return bce

class TotalLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.boundarymap = BoundaryMapLoss()
        self.bregular = BoundaryRegular()
        self.sregular = SegmentationRegular()
    
    def forward(self,input,edge,target,edge_label,alpha1 = 0., alpha2 = 0.):

        bce = F.binary_cross_entropy_with_logits(input, target)

        edge_loss = self.boundarymap(edge,edge_label)
        # print(input.size())
        # print(edge.size())
        bregular = self.bregular(input,edge)

        sregular = self.sregular(input,edge)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        

        return 0.5*bce + dice + 0.2*edge_loss, edge_loss, bregular, sregular


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

def CE_Loss(inputs, target, num_classes=1):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
    return CE_loss


