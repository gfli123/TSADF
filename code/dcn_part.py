import torch

from modules.modulated_deform_conv import _ModulatedDeformConv
from modules.modulated_deform_conv import ModulatedDeformConvPack

class DCNv2Pack(ModulatedDeformConvPack):
    """
       This source file is originally from: https://github.com/ZhaoHengrun/EDVR
    """

    def forward(self, x, feat):
        out = self.conv_offset_mask(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return _ModulatedDeformConv(x, offset, mask, self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups, self.deformable_groups,
                                    self.im2col_step)