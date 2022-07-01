import torch
from torch import nn
from torch.nn import Conv2d, Conv3d
import torch.nn.functional as F

class spatial_channel_attention(nn.Module):
    def __init__(self):
        super(spatial_channel_attention, self).__init__()
        # self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_c1 = nn.Conv2d(192, 192//16, (1, 1), bias=False)
        self.conv_c2 = nn.Conv2d(192//16,192, (1, 1), bias=False)
        self.conv_s1 = nn.Conv2d(192, 192, (1, 1), bias=False)
        self.conv_s2 = nn.Conv2d(192, 192, (1, 1), groups=192,bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        '''

        :param inputs: Modules that need to apply the attention mechanism. Size (B, C, H, W）
        :return: Results of fusion after application of attentional mechanism. Size (B, C, H, W）
        '''
        ca = torch.nn.functional.adaptive_avg_pool2d(inputs, (1,1))
        ca = self.conv_c1(ca)                   # ca channel attention
        ca = self.lrelu(ca)
        ca = self.conv_c2(ca)

        sa = self.conv_s1(inputs)               # sa: spatial attention
        sa = self.lrelu(sa)
        sa = self.conv_s2(sa)

        spatial_channel_fea = self.sigmoid(sa + ca)
        outputs_attention = torch.mul(spatial_channel_fea, inputs)
        outputs = outputs_attention + inputs

        return outputs

class Res_Block2D(nn.Module):
    '''Residual block w/o BN
        ---Conv-ReLU-Conv-+-
         |________________|
    '''
    def __init__(self, nf):
        super(Res_Block2D, self).__init__()
        self.conv1 = Conv2d(nf, nf, (3, 3), stride=(1,), padding=(1,), groups=4)
        self.conv2 = Conv2d(nf, nf, (3, 3), stride=(1,), padding=(1,), groups=4)
    def forward(self, inputs):
        identity = inputs
        x = F.relu(self.conv1(inputs), inplace=True)
        x = self.conv2(x)
        outputs = F.relu(identity + x)
        return outputs

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    print(x.size()[-2:])
    print(flow.size()[1:3])
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
