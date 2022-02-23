import torch
from torch import nn


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx, pairwise_distance


def local_operator(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1).contiguous()[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims).contiguous()

    x = x.view(batch_size, num_points, 1, num_dims).contiguous().repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor), dim=3).permute(0, 3, 1, 2).contiguous()  # local and global all in

    return feature


def local_operator_withnorm(x, norm_plt, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    norm_plt = norm_plt.view(batch_size, -1, num_points)
    idx, _ = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    norm_plt = norm_plt.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = norm_plt.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor-x, neighbor, neighbor_norm), dim=3).permute(0, 3, 1, 2)  # 3c

    return feature


def GDM(x, M):
    """
    Geometry-Disentangle Module
    M: number of disentangled points in both sharp and gentle variation components
    """
    k = 64  # number of neighbors to decide the range of j in Eq.(5)
    tau = 0.2  # threshold in Eq.(2)
    sigma = 2  # parameters of f (Gaussian function in Eq.(2))
    ###############
    """Graph Construction:"""
    device = torch.device('cuda')
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx, p = knn(x, k=k)  # p: -[(x1-x2)^2+...]

    # here we add a tau
    p1 = torch.abs(p)
    p1 = torch.sqrt(p1)
    mask = p1 < tau

    # here we add a sigma
    p = p / (sigma * sigma)
    w = torch.exp(p)  # b,n,n
    w = torch.mul(mask.float(), w)

    b = 1/torch.sum(w, dim=1)
    b = b.reshape(batch_size, num_points, 1).repeat(1, 1, num_points)
    c = torch.eye(num_points, num_points, device=device)
    c = c.expand(batch_size, num_points, num_points)
    D = b * c  # b,n,n

    A = torch.matmul(D, w)  # normalized adjacency matrix A_hat

    # Get Aij in a local area:
    idx2 = idx.view(batch_size * num_points, -1)
    idx_base2 = torch.arange(0, batch_size * num_points, device=device).view(-1, 1) * num_points
    idx2 = idx2 + idx_base2

    idx2 = idx2.reshape(batch_size * num_points, k)[:, 1:k]
    idx2 = idx2.reshape(batch_size * num_points * (k - 1))
    idx2 = idx2.view(-1)

    A = A.view(-1).contiguous()
    A = A[idx2].reshape(batch_size, num_points, k - 1).contiguous()  # Aij: b,n,k
    ###############
    """Disentangling Point Clouds into Sharp(xs) and Gentle(xg) Variation Components:"""
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(batch_size * num_points, k)[:, 1:k]
    idx = idx.reshape(batch_size * num_points * (k - 1))

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # b,n,c
    neighbor = x.view(batch_size * num_points, -1).contiguous()[idx, :]
    neighbor = neighbor.view(batch_size, num_points, k - 1, num_dims).contiguous()  # b,n,k,c
    A = A.reshape(batch_size, num_points, k - 1, 1).contiguous()  # b,n,k,1
    n = A.mul(neighbor)  # b,n,k,c
    n = torch.sum(n, dim=2)  # b,n,c

    pai = torch.norm(x - n, dim=-1).pow(2)  # Eq.(5)
    pais = pai.topk(k=M, dim=-1)[1]  # first M points as the sharp variation component
    paig = (-pai).topk(k=M, dim=-1)[1]  # last M points as the gentle variation component

    pai_base = torch.arange(0, batch_size, device=device).view(-1, 1) * num_points
    indices = (pais + pai_base).view(-1)
    indiceg = (paig + pai_base).view(-1)

    xs = x.view(batch_size * num_points, -1).contiguous()[indices, :]
    xg = x.view(batch_size * num_points, -1).contiguous()[indiceg, :]

    xs = xs.view(batch_size, M, -1).contiguous()  # b,M,c
    xg = xg.view(batch_size, M, -1).contiguous()  # b,M,c

    return xs, xg


class SGCAM(nn.Module):
    """Sharp-Gentle Complementary Attention Module:"""
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(SGCAM, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_2):
        batch_size = x.size(0)

        g_x = self.g(x_2).view(batch_size, self.inter_channels, -1).contiguous()
        g_x = g_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).contiguous()
        theta_x = theta_x.permute(0, 2, 1).contiguous()
        phi_x = self.phi(x_2).view(batch_size, self.inter_channels, -1).contiguous()
        W = torch.matmul(theta_x, phi_x)  # Attention Matrix
        N = W.size(-1)
        W_div_C = W / N

        y = torch.matmul(W_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]).contiguous()
        W_y = self.W(y)
        y = W_y + x

        return y

