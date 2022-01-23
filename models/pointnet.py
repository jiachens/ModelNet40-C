# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model import PointNetCls
from all_utils import DATASET_NUM_CLASS

class PointNet(nn.Module):

    def __init__(self, dataset, task):
        super().__init__()
        self.task = task
        num_class = DATASET_NUM_CLASS[dataset]
        if self.task == 'cls_trans':
            self.model =  PointNetCls(k=num_class, feature_transform=True)
        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.transpose(2, 1).float()
        if self.task == 'cls_trans':
            logit, _, trans_feat = self.model(pc)
        else:
            assert False

        out = {'logit': logit, 'trans_feat': trans_feat}
        return out
