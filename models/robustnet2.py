import torch.nn as nn
import torch.nn.functional as F
from all_utils import DATASET_NUM_CLASS
from CurveNet.core.models.curvenet_util import *
from GDANet.model.util.GDANet_util import local_operator, GDM, SGCAM

class RobustNet2(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            self.model = RobustNet2_OG(num_classes=num_classes)

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

curve_config = {
        'default': [[100, 5], [100, 5], None, None],
        'long':  [[10, 30], None,  None,  None]
    }

class RobustNet2_OG(nn.Module):
    def __init__(self, num_classes=40, k=20, setting='default'):
        super(RobustNet2_OG, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic1 = CIC(npoint=1024,radius=0.05, k=k,  in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=1024, radius=0.05, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][0])

        self.cic2 = CIC(npoint=512,radius=0.15, k=k,  in_channels=64, output_channels=128, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=512,radius=0.3, k=k,  in_channels=128, output_channels=128, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][1])
        
        self.cic3 = CIC(npoint=256,radius=0.2, k=k,  in_channels=128, output_channels=256, bottleneck_ratio=2, mlp_num=1, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=256,radius=0.4, k=k,  in_channels=256, output_channels=256, bottleneck_ratio=4, mlp_num=1, curve_config=curve_config[setting][2])

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(512),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.SGCAM_1s = SGCAM(512)
        self.SGCAM_1g = SGCAM(512)
        # self.conv1 = nn.Sequential(nn.Conv1d(128 * 2, 64, kernel_size=1, bias=True),
        #                             self.bn12)
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)

        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        batch_size, _, _ = xyz.size()

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic1(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic2(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic3(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
 
        # l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        # l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.pt_last(l3_points)
        x = torch.cat([x, l3_points], dim=1)
        x = self.conv_fuse(x)

        x1s, x1g = GDM(x, M=128)
        y1s = self.SGCAM_1s(x, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x, x1g.transpose(2, 1))

        x = torch.cat([y1s, y1g], 1)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        x = self.linear3(x)

        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        # batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
