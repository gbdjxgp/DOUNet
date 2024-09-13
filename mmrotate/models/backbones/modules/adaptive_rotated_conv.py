import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import grid_sample

__all__ = ['AdaptiveRotatedConv2d','rotate_conv_kernel']

def rotate_conv_kernel(conv_weights,lambdas,thetas, padding='reflection'):
    # 输入：conv_weights:卷积核的权重，形状[k_n,C_out,C_in,k(H),k(W)]
    # lambdas=thetas=[bs,k_n]
    device = conv_weights.device
    bs,k_n = thetas.shape
    _, Cout, Cin, h, w = conv_weights.shape
    assert _ == k_n
    # 创建一个2x2的旋转矩阵 rotation_matrix，用来描述旋转操作
    # thetas:[bs, k_n]->[bs, k_n,1,1]->rotation_matrix:[bs, k_n,2,2]
    thetas = thetas.unsqueeze(-1).unsqueeze(-1)
    cosa = torch.cos(thetas)
    sina = torch.sin(thetas)
    rotation_matrix = torch.cat([torch.cat([cosa, -sina], dim=-1), torch.cat([sina, cosa], dim=-1)], dim=-2)

    # 创建grid，代表原来卷积的一系列坐标点,使用mul进行旋转变换
    # [h,w,2]->[bs,k_n,h,w,2]->(bs*k_n,h*w,2).mul(bs*k_n,2,2)->[bs*k_n,h*w,2]->[bs*k_n,h,w,2]
    x_range = torch.linspace(-1, 1, w, device=device)
    y_range = torch.linspace(-1, 1, h, device=device)
    # 使用这些坐标范围创建一个网格，其中y表示纵向坐标，x表示横向坐标
    y, x = torch.meshgrid(y_range, x_range)
    # 将 x 和 y 合并成一个网格张量 grid，进行扩展
    grid = torch.stack([x, y], -1).expand([bs,k_n, -1, -1, -1])

    # 将网格张量重塑为二维形状，并应用旋转矩阵，然后重新将其形状调整
    grid = grid.reshape(-1,h*w,2).matmul(rotation_matrix.reshape(-1,2,2)).view(bs*k_n,h,w,2)
    # 使用双线性插值方法在变换后的网格上对输入图像进行采样，从而实现图像的旋转操作
    # 卷积权重的操作conv_weights。[k_n,C_out,C_in,k(H),k(W)]->[bs,k_n,C_out,C_in,k(H),k(W)]->[bs*k_n,cout*cin,h,w]
    conv_weights = conv_weights.expand([bs, k_n,Cout,Cin,h,w]).reshape([-1,  Cout*Cin, h, w])
    # 采样操作
    conv_weights = grid_sample(conv_weights, grid, 'bilinear', padding, align_corners=True).view(bs,k_n,Cout,Cin,h,w)
    # 采样之后：[bs*k_n,cout*cin,h,w]->[bs,k_n,cout,cin,h,w]
    # lambdas：[bs,k_n]->[bs,k_n,1,1,1,1]
    conv_weights = (lambdas.reshape(bs,k_n, 1, 1, 1, 1) * conv_weights).sum(dim=1).view(bs*Cout,Cin,h,w)
    return conv_weights


def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    device = thetas.device
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1-a, torch.zeros(1, 7, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), x-b, b, torch.zeros(1, 1, bs*g, device=device), 1-c+b, y-b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device), 1-a, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((b, y-b, torch.zeros(1,1 , bs*g, device=device), x-b, 1-c+b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-c+b, x-b, torch.zeros(1, 1, bs*g, device=device), y-b, b), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), 1-a, torch.zeros(1, 2, bs*g, device=device), a, torch.zeros(1, 2, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), y-b, 1-c+b, torch.zeros(1, 1, bs*g, device=device), b, x-b, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 7, bs*g, device=device), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, torch.zeros(1, 2, bs*g, device=device), 1-c, torch.zeros(1, 5, bs*g, device=device)), dim=1),
        torch.cat((-b, x+b, torch.zeros(1, 1, bs*g, device=device), b-y, 1-a-b, torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), 1-c, c, torch.zeros(1, 6, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 3, bs*g, device=device), x+b, 1-a-b, torch.zeros(1, 1, bs*g, device=device), -b, b-y, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), torch.ones(1, 1, bs*g, device=device), torch.zeros(1, 4, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 1, bs*g, device=device), b-y, -b, torch.zeros(1, 1, bs*g, device=device), 1-a-b, x+b, torch.zeros(1, 3, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 6, bs*g, device=device), c, 1-c, torch.zeros(1, 1, bs*g, device=device)), dim=1),
        torch.cat((torch.zeros(1, 4, bs*g, device=device), 1-a-b, b-y, torch.zeros(1, 1, bs*g, device=device), x+b, -b), dim=1),
        torch.cat((torch.zeros(1, 5, bs*g, device=device), 1-c, torch.zeros(1, 2, bs*g, device=device), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.float()                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat


def batch_rotate_multiweight(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(thetas.shape == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas)

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape 
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.contiguous().view(n*9, Cout*Cin)


    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights

class AdaptiveRotatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, groups=1, bias=False,
                 kernel_number=1, rounting_func=None, rotate_func=batch_rotate_multiweight):
        super().__init__()
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.rounting_func = rounting_func
        self.rotate_func = rotate_func

        self.weight = nn.Parameter(
            torch.Tensor(
                kernel_number, 
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # get alphas, angles
        # # [bs, Cin, h, w] --> [bs, n_theta], [bs, n_theta]
        alphas, angles = self.rounting_func(x)

        # rotate weight
        # # [Cout, Cin, k, k] --> [bs * Cout, Cin, k, k]
        rotated_weight = self.rotate_func(self.weight, alphas, angles)

        # reshape images
        bs, Cin, h, w = x.shape
        x = x.reshape(1, bs * Cin, h, w)  # [1, bs * Cin, h, w]
        
        # adaptive conv over images using group conv
        out = F.conv2d(input=x, weight=rotated_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=(self.groups * bs))
        
        # reshape back
        out = out.reshape(bs, self.out_channels, *out.shape[2:])
        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_number={kernel_number}'
             ', kernel_size={kernel_size}, stride={stride}, bias={bias}')
             
        if self.padding != (0,) * len([self.padding]):
            s += ', padding={padding}'
        if self.dilation != (1,) * len([self.dilation]):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
    