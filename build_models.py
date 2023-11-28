import torch
import math
import torch.nn.functional as F
import numpy as np
from option import opt
from torch import nn
from torch.nn import Conv2d, Conv3d, Sequential
from utils import Res_Block2D, UPSAMPLE, Res_Block2D_Add
from option import opt
from dcn_part import DCNv2Pack


class Add1(nn.Module):
    def __init__(self, nf, act):
        super(Add1, self).__init__()
        self.nf = nf
        self.hf = self.nf // 2
        self.act = act

        self.encoder = Sequential(Conv2d(self.nf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                                  Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                                  Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act)

        self.mu = Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.logvar = Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.decoder = Sequential(Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                                  Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), self.act,
                                  Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1)), nn.Sigmoid())


    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu, requires_grad=True)
        z = mu + eps * torch.exp(logvar/2)

        return z

    def forward(self, ref):
        fea = self.encoder(ref)       
        mu = self.mu(fea)
        logvar = self.logvar(fea)
        z = self.reparameterize(mu, logvar)
        recons = self.decoder(z)

        return recons


class FeaExt(nn.Module):
    def __init__(self, nf, hf, num_b1, idx, act):
        super(FeaExt, self).__init__()
        self.nf = nf
        self.hf = hf
        self.num_b1 = num_b1
        self.idx = idx
        self.act = act
        self.fea = Res_Block2D_Add(3, self.nf, self.num_b1, self.act)
        self.add = Add1(self.nf, self.act)
        self.fus = Conv2d(self.nf * 3 // 2, self.hf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.pre = Conv2d(self.hf * (self.idx + 1), self.hf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.ref = Conv2d(self.hf, self.hf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.post = Conv2d(self.hf * (self.idx + 1), self.hf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, inputs):
        b, t, c, h, w = inputs.shape
        fea = self.fea(inputs.reshape(b * t, c, h, w))
        add = self.add(fea)
        fus = self.act(self.fus(torch.cat((fea, add), dim=1))).reshape(b, t, self.hf, h, w)

        img_pre = fus[:, :self.idx + 1, :, :, :]
        img_ref = fus[:, self.idx, :, :, :]
        img_post = fus[:, self.idx:, :, :, :]
        
        img_pre = self.act(self.pre(img_pre.reshape(b, self.hf * (self.idx + 1), h, w)))
        img_ref = self.act(self.ref(img_ref))
        img_post = self.act(self.post(img_post.reshape(b, self.hf * (self.idx + 1), h, w)))

        return img_pre, img_ref, img_post


class MultiPCAligned(nn.Module):
    def __init__(self, nf, act, deformable_groups=8):
        super(MultiPCAligned, self).__init__()
        self.nf = nf
        self.act = act
        self.scale = 2
        self.down_ref1 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.down_ref2 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.down_other1 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.down_other2 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))

        self.L3_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_offset2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_dif1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_dif2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L3_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L2_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_dif1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_dif2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L2_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_fuseL3 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L1_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_dif1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_dif2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L1_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_fuseL2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, ref, other):
        ref_down1 = self.act(self.down_ref1(ref))
        ref_down2 = self.act(self.down_ref2(ref_down1))
        other_down1 = self.act(self.down_other1(other))
        other_down2 = self.act(self.down_other2(other_down1))

        l3 = torch.cat([ref_down2, other_down2], dim=1)
        l3_offset = self.act(self.L3_offset1(l3))
        l3_offset = self.act(self.L3_offset2(l3_offset))
        l3_dif = ref_down2 - other_down2
        l3_dif = self.act(self.L3_dif1(l3_dif))
        l3_dif = self.act(self.L3_dif2(l3_dif))
        l3_dcn = self.L3_dcn(other_down2, l3_offset)
        l3_offset = F.interpolate(l3_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l3_fuse = self.act(self.L3_fuse(torch.cat([l3_dif, l3_dcn], dim=1)))
        l3_fuse = F.interpolate(l3_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l2 = torch.cat([ref_down1, other_down1], dim=1)
        l2_offset = self.act(self.L2_offset1(l2))
        l2_offset = self.act(self.L2_offset2(torch.cat([l2_offset, l3_offset * 2], dim=1)))
        l2_offset = self.act(self.L2_offset3(l2_offset))
        l2_dif = ref_down1 - other_down1
        l2_dif = self.act(self.L2_dif1(l2_dif))
        l2_dif = self.act(self.L2_dif2(l2_dif))
        l2_dcn = self.L2_dcn(other_down1, l2_offset)
        l2_offset = F.interpolate(l2_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l2_fuse = self.act(self.L2_fuse(torch.cat([l2_dif, l2_dcn], dim=1)))
        l2_fuse = self.act(self.L2_fuseL3(torch.cat([l2_fuse, l3_fuse], dim=1)))
        l2_fuse = F.interpolate(l2_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l1 = torch.cat([ref, other], dim=1)
        l1_offset = self.act(self.L1_offset1(l1))
        l1_offset = self.act(self.L1_offset2(torch.cat([l1_offset, l2_offset * 2], dim=1)))
        l1_offset = self.act(self.L1_offset3(l1_offset))
        l1_dif = ref - other
        l1_dif = self.act(self.L1_dif1(l1_dif))
        l1_dif = self.act(self.L1_dif2(l1_dif))
        l1_dcn = self.L1_dcn(other, l1_offset)
        l1_fuse = self.act(self.L1_fuse(torch.cat([l1_dif, l1_dcn], dim=1)))
        l1_fuse = self.act(self.L1_fuseL2(torch.cat([l1_fuse, l2_fuse], dim=1)))

        return l1_fuse


class VSR(nn.Module):
    def __init__(self, nf=64, hf=64, num_b1=5, num_b2=10):
        super(VSR, self).__init__()
        self.nf = nf
        self.hf = hf
        self.num_inputs = opt.num_frames
        self.num_b1 = num_b1
        self.num_b2 = num_b2
        self.scale = opt.scale
        self.idx = self.num_inputs // 2
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.fea_ext = FeaExt(self.nf, self.hf, self.num_b1, self.idx, self.act)
        self.multi_PC = MultiPCAligned(self.hf, self.act)

        self.recons1 = Res_Block2D_Add(3 * self.hf, self.hf, self.num_b2, self.act)
        self.recons2 = Res_Block2D_Add(4 * self.hf, self.hf, self.num_b2, self.act)

        self.out = UPSAMPLE(self.hf, self.act)

    def forward(self, inputs):
        img_ref_bic = F.interpolate(inputs[:, self.idx, :, :, :], scale_factor=self.scale, mode='bicubic', align_corners=False)
        pre, ref, post = self.fea_ext(inputs)
        pre_align = self.multi_PC(ref, pre)
        post_align = self.multi_PC(ref, post)

        recons = self.recons1(torch.cat((pre_align, post_align, ref), dim=1))
        pre_dif = ref - pre_align
        post_dif = ref - post_align
        recons = self.recons2(torch.cat((recons, pre_dif, post_dif, ref), dim=1))
        recons = self.out(recons)
        out = recons + img_ref_bic

        return out

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)
        if isinstance(m, Conv3d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)


