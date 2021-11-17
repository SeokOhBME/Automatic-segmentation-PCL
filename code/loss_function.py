import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        smooth = 1e-5
        num = targets.size(0)
        inputs = inputs.view(num, -1)
        target = targets.view(num, -1)
        intersection = (inputs * target)
        dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        return dice

import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input, target):


        loss = nn.BCELoss()
        bce = loss(input, target)
        smooth = 1e-5
        input = input
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return (bce) + dice
