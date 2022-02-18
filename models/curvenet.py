'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-17 20:37:07
LastEditors: Jiachen Sun
LastEditTime: 2022-02-17 20:42:20
'''
import torch.nn as nn
import torch.nn.functional as F
from CurveNet.core.models.curvenet_cls import CurveNet as CurveNet_og
from all_utils import DATASET_NUM_CLASS

class CurveNet(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            self.model = CurveNet_og(num_classes=num_classes)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
