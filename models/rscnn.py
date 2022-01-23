import torch
import torch.nn as nn
import torch.nn.functional as F
from rs_cnn.models import RSCNN_MSN_Seg, RSCNN_SSN_Cls
from all_utils import DATASET_NUM_CLASS
# distilled from:
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/models/rscnn_ssn_cls.py
# https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/models/rscnn_msn_seg.py
class RSCNN(nn.Module):

    def __init__(self, task, dataset, ssn_or_msn):
        """
        Returns a model
        :param cls_or_seg: (bool) if true cls else seg
        :param ssn_or_msn: (bool) if true ssn else msn
        """
        super().__init__()
        self.task = task
        self.dataset = dataset
        num_classes = DATASET_NUM_CLASS[self.dataset]
        if self.task == 'cls':
            assert ssn_or_msn
            # source: https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/cfgs/config_ssn_cls.yaml
            # source: https://github.com/Yochengliu/Relation-Shape-CNN/blob/master/train_cls.py#L73
            rscnn_params = {
                'num_classes':num_classes,
                'input_channels': 0,
                'relation_prior': 1,
                'use_xyz': True
            }
            self.model = RSCNN_SSN_Cls(**rscnn_params)
        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        if self.task == 'cls':
            assert cls is None
            out = {'logit': self.model(pc)}
        else:
            assert False
        return out
