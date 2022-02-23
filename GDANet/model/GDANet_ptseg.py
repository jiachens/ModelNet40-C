import torch.nn as nn
import torch
import torch.nn.functional as F
from util.GDANet_util import local_operator_withnorm, local_operator, GDM, SGCAM


class GDANet(nn.Module):
    def __init__(self, num_classes):
        super(GDANet, self).__init__()

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn5 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(128, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)
        self.convc = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=True),
                                   self.bnc)

        self.conv5 = nn.Sequential(nn.Conv1d(256 + 512 + 64, 256, kernel_size=1, bias=True),
                                   self.bn5)
        self.dp1 = nn.Dropout(0.4)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=True),
                                   self.bn6)
        self.dp2 = nn.Dropout(0.4)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=True),
                                   self.bn7)
        self.conv8 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

    def forward(self, x, norm_plt, cls_label):
        B, C, N = x.size()
        ###############
        """block 1"""
        x1 = local_operator_withnorm(x, norm_plt, k=30)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]
        x1h, x1l = GDM(x1, M=512)

        x1h = self.SGCAM_1s(x1, x1h.transpose(2, 1))
        x1l = self.SGCAM_1g(x1, x1l.transpose(2, 1))
        x1 = torch.cat([x1h, x1l], 1)
        x1 = F.relu(self.conv12(x1))
        ###############
        """block 1"""
        x1t = torch.cat((x, x1), dim=1)
        x2 = local_operator(x1t, k=30)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]
        x2h, x2l = GDM(x2, M=512)

        x2h = self.SGCAM_2s(x2, x2h.transpose(2, 1))
        x2l = self.SGCAM_2g(x2, x2l.transpose(2, 1))
        x2 = torch.cat([x2h, x2l], 1)
        x2 = F.relu(self.conv22(x2))
        ###############
        x2t = torch.cat((x1t, x2), dim=1)
        x3 = local_operator(x2t, k=30)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        x3 = F.relu(self.conv32(x3))
        ###############
        xx = torch.cat((x1, x2, x3), dim=1)

        xc = F.relu(self.conv4(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)

        cls_label = cls_label.view(B, 16, 1)
        cls_label = F.relu(self.convc(cls_label))
        cls = torch.cat((xc.view(B, 512, 1), cls_label), dim=1)
        cls = cls.repeat(1, 1, N)

        x = torch.cat((xx, cls), dim=1)
        x = F.relu(self.conv5(x))
        x = self.dp1(x)
        x = F.relu(self.conv6(x))
        x = self.dp2(x)
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # b,n,50

        return x

