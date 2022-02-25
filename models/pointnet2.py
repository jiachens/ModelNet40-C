'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-16 22:23:16
LastEditors: Jiachen Sun
LastEditTime: 2022-02-24 22:36:59
'''
import torch
import torch.nn as nn
from pointnet2_pyt.pointnet2.models.pointnet2_msg_cls import Pointnet2MSG
from all_utils import DATASET_NUM_CLASS

class PointNet2(nn.Module):

    def __init__(self, task, dataset, version_cls):
        super().__init__()
        self.task =  task
        num_class = DATASET_NUM_CLASS[dataset]
        if task == 'cls':
            self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True, version=version_cls)
        else:
            assert False

    def forward(self, pc, normal=None, cls=None):
        pc = pc.to(next(self.parameters()).device)
        if self.task == 'cls':
            assert cls is None
            assert normal is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False
        return out
