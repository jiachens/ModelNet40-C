"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_normal.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from .curvenet_util import *


curve_config = {
        'default': [[100, 5], [100, 5], None, None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=3, k=20, multiplier=1.0, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 64
        channels = [128, 256, 512, 1024]
        channels = [int(c * multiplier) for c in channels]
        
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=1024, radius=0.1, k=k, in_channels=additional_channel, output_channels=channels[0], bottleneck_ratio=2, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.1, k=k, in_channels=channels[0], output_channels=channels[0], bottleneck_ratio=4, curve_config=curve_config[setting][0])
        
        self.cic21 = CIC(npoint=256, radius=0.2, k=k, in_channels=channels[0], output_channels=channels[1], bottleneck_ratio=2, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=256, radius=0.2, k=k, in_channels=channels[1], output_channels=channels[1], bottleneck_ratio=4, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=64, radius=0.4, k=k, in_channels=channels[1], output_channels=channels[2], bottleneck_ratio=2, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=64, radius=0.4, k=k, in_channels=channels[2], output_channels=channels[2], bottleneck_ratio=4, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=16, radius=0.8, k=15, in_channels=channels[2], output_channels=channels[3], bottleneck_ratio=2, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=16, radius=0.8, k=15, in_channels=channels[3], output_channels=channels[3], bottleneck_ratio=4, curve_config=curve_config[setting][3])
        #self.cic43 = CIC(npoint=16, radius=0.8, k=15, in_channels=2048, output_channels=2048, bottleneck_ratio=4, curve_config=curve_config[setting][3])
        # decoder
        self.fp3 = PointNetFeaturePropagation(in_channel=channels[3] + channels[2], mlp=[channels[2], channels[2]], att=[channels[3], channels[3]//2, channels[3]//8])
        self.up_cic4 = CIC(npoint=64, radius=0.8, k=k, in_channels=channels[2], output_channels=channels[2], bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation(in_channel=channels[2] + channels[1], mlp=[channels[1], channels[1]], att=[channels[2], channels[2]//2, channels[2]//8])
        self.up_cic3 = CIC(npoint=256, radius=0.4, k=k, in_channels=channels[1], output_channels=channels[1], bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation(in_channel=channels[1] + channels[0], mlp=[channels[0], channels[0]], att=[channels[1], channels[1]//2, channels[1]//8])
        self.up_cic2 = CIC(npoint=1024, radius=0.1, k=k, in_channels=channels[0]+3, output_channels=channels[0], bottleneck_ratio=4)
        self.up_cic1 = CIC(npoint=1024, radius=0.1, k=k, in_channels=channels[0], output_channels=channels[0], bottleneck_ratio=4)

        self.point_conv = nn.Sequential(
            nn.Conv2d(9, additional_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(additional_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Conv1d(channels[0], num_classes, 1)

    def forward(self, xyz):
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)
        #l4_xyz, l4_points = self.cic43(l4_xyz, l4_points)

        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_xyz, l3_points = self.up_cic4(l3_xyz, l3_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_xyz, l2_points = self.up_cic3(l2_xyz, l2_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        x = torch.cat((l1_xyz, l1_points), dim=1)

        xyz, x = self.up_cic2(l1_xyz, x)
        xyz, x = self.up_cic1(xyz, x)

        x = self.conv1(x)
        return x
