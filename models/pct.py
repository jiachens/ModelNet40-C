import torch.nn as nn
from PCT_Pytorch.model import Pct as Pct_original
from all_utils import DATASET_NUM_CLASS

class Pct(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pct_original(args, output_channels=num_classes)

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
