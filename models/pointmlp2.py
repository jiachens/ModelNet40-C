import torch.nn as nn
from pointMLP.classification_ModelNet40.models.pointmlp import pointMLPElite as pointMLP_original
from all_utils import DATASET_NUM_CLASS

class pointMLP2(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            self.model = pointMLP_original(num_classes=num_classes)

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
