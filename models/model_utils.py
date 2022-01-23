import torch.nn as nn
# from syncbn_pyt.modules.nn import BatchNorm2d as BatchNorm2dSync

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.squeeze()

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)

