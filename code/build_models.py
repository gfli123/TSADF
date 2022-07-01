import torch
import torch.nn.functional as F
from option import opt
from torch import nn
from torch.nn import Conv2d, Sequential, Conv3d
from module_utils import Res_Block2D
from dcn_part import DCNv2Pack


class FeaExt(nn.Module):
    def __init__(self, nf):
        super(FeaExt, self).__init__()
        self.conv1 = Conv3d(3, nf, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv2 = Conv3d(nf, nf, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = Conv3d(nf, nf, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.pre = Conv3d(nf, nf, (3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
        self.ref = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.post = Conv3d(nf, nf, (3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, inputs, idx):
        x = self.lrelu(self.conv1(inputs))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        img_pre = x[:, :, :idx, :, :]
        img_ref = x[:, :, idx, :, :]
        img_post = x[:, :, idx + 1:, :, :]
        img_pre = self.lrelu(self.pre(img_pre)).squeeze(2)
        img_ref = self.lrelu(self.ref(img_ref))
        img_post = self.lrelu(self.post(img_post)).squeeze(2)

        return img_pre, img_ref, img_post


class MultiPCAligned(nn.Module):
    def __init__(self, nf, deformable_groups=8):
        super(MultiPCAligned, self).__init__()
        self.scale = 2
        self.pre_downsample1 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.pre_downsample2 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.ref_downsample1 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.ref_downsample2 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.post_downsample1 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))
        self.post_downsample2 = Conv2d(nf, nf, (3, 3), stride=(2, 2), padding=(1, 1))

        self.L3_pre_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_pre_offset2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_pre_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_pre_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_pre_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L3_pre_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L2_pre_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L2_pre_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_pre_fuseL3 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L1_pre_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L1_pre_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_pre_fuseL2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L3_post_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_post_offset2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_post_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_post_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L3_post_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L3_post_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L2_post_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L2_post_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L2_post_fuseL3 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.L1_post_offset1 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_offset2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_offset3 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_max1 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_max2 = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_dcn = DCNv2Pack(nf, nf, 3, stride=1, padding=1, deformable_groups=deformable_groups)
        self.L1_post_fuse = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.L1_post_fuseL2 = Conv2d(nf * 2, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, pre, img_ref, post):
        img_pre_down1 = self.lrelu(self.pre_downsample1(pre))
        img_pre_down2 = self.lrelu(self.pre_downsample2(img_pre_down1))
        img_ref_down1 = self.lrelu(self.ref_downsample1(img_ref))
        img_ref_down2 = self.lrelu(self.ref_downsample2(img_ref_down1))
        img_post_down1 = self.lrelu(self.post_downsample1(post))
        img_post_down2 = self.lrelu(self.post_downsample2(img_post_down1))

        l3_pre = torch.cat([img_pre_down2, img_ref_down2], dim=1)
        l3_pre_offset = self.lrelu(self.L3_pre_offset1(l3_pre))
        l3_pre_offset = self.lrelu(self.L3_pre_offset2(l3_pre_offset))
        l3_pre_max = img_ref_down2 - img_pre_down2
        l3_pre_max = self.lrelu(self.L3_pre_max1(l3_pre_max))
        l3_pre_max = self.lrelu(self.L3_pre_max2(l3_pre_max))
        l3_pre_dcn = self.L3_pre_dcn(img_pre_down2, l3_pre_offset)
        l3_pre_offset = F.interpolate(l3_pre_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l3_pre_fuse = self.lrelu(self.L3_pre_fuse(torch.cat([l3_pre_max, l3_pre_dcn], dim=1)))
        l3_pre_fuse = F.interpolate(l3_pre_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l2_pre = torch.cat([img_pre_down1, img_ref_down1], dim=1)
        l2_pre_offset = self.lrelu(self.L2_pre_offset1(l2_pre))
        l2_pre_offset = self.lrelu(self.L2_pre_offset2(torch.cat([l2_pre_offset, l3_pre_offset * 2], dim=1)))
        l2_pre_offset = self.lrelu(self.L2_pre_offset3(l2_pre_offset))
        l2_pre_max = img_ref_down1 - img_pre_down1
        l2_pre_max = self.lrelu(self.L2_pre_max1(l2_pre_max))
        l2_pre_max = self.lrelu(self.L2_pre_max2(l2_pre_max))
        l2_pre_dcn = self.L2_pre_dcn(img_pre_down1, l2_pre_offset)
        l2_pre_offset = F.interpolate(l2_pre_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l2_pre_fuse = self.lrelu(self.L2_pre_fuse(torch.cat([l2_pre_max, l2_pre_dcn], dim=1)))
        l2_pre_fuse = self.lrelu(self.L2_pre_fuseL3(torch.cat([l2_pre_fuse, l3_pre_fuse], dim=1)))
        l2_pre_fuse = F.interpolate(l2_pre_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l1_pre = torch.cat([pre, img_ref], dim=1)
        l1_pre_offset = self.lrelu(self.L1_pre_offset1(l1_pre))
        l1_pre_offset = self.lrelu(self.L1_pre_offset2(torch.cat([l1_pre_offset, l2_pre_offset * 2], dim=1)))
        l1_pre_offset = self.lrelu(self.L1_pre_offset3(l1_pre_offset))
        l1_pre_max = img_ref - pre
        l1_pre_max = self.lrelu(self.L1_pre_max1(l1_pre_max))
        l1_pre_max = self.lrelu(self.L1_pre_max2(l1_pre_max))
        l1_pre_dcn = self.L1_pre_dcn(pre, l1_pre_offset)
        l1_pre_fuse = self.lrelu(self.L1_pre_fuse(torch.cat([l1_pre_max, l1_pre_dcn], dim=1)))
        l1_pre_fuse = self.lrelu(self.L1_pre_fuseL2(torch.cat([l1_pre_fuse, l2_pre_fuse], dim=1)))

        l3_post = torch.cat([img_post_down2, img_ref_down2], dim=1)
        l3_post_offset = self.lrelu(self.L3_post_offset1(l3_post))
        l3_post_offset = self.lrelu(self.L3_post_offset2(l3_post_offset))
        l3_post_max = img_ref_down2 - img_post_down2
        l3_post_max = self.lrelu(self.L3_post_max1(l3_post_max))
        l3_post_max = self.lrelu(self.L3_post_max2(l3_post_max))
        l3_post_dcn = self.L3_post_dcn(img_post_down2, l3_post_offset)
        l3_post_offset = F.interpolate(l3_post_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l3_post_fuse = self.lrelu(self.L3_post_fuse(torch.cat([l3_post_max, l3_post_dcn], dim=1)))
        l3_post_fuse = F.interpolate(l3_post_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l2_post = torch.cat([img_post_down1, img_ref_down1], dim=1)
        l2_post_offset = self.lrelu(self.L2_post_offset1(l2_post))
        l2_post_offset = self.lrelu(self.L2_post_offset2(torch.cat([l2_post_offset, l3_post_offset * 2], dim=1)))
        l2_post_offset = self.lrelu(self.L2_post_offset3(l2_post_offset))
        l2_post_max = img_ref_down1 - img_post_down1
        l2_post_max = self.lrelu(self.L2_post_max1(l2_post_max))
        l2_post_max = self.lrelu(self.L2_post_max2(l2_post_max))
        l2_post_dcn = self.L2_post_dcn(img_post_down1, l2_post_offset)
        l2_post_offset = F.interpolate(l2_post_offset, scale_factor=self.scale, mode='bilinear', align_corners=False)
        l2_post_fuse = self.lrelu(self.L2_post_fuse(torch.cat([l2_post_max, l2_post_dcn], dim=1)))
        l2_post_fuse = self.lrelu(self.L2_post_fuseL3(torch.cat([l2_post_fuse, l3_post_fuse], dim=1)))
        l2_post_fuse = F.interpolate(l2_post_fuse, scale_factor=self.scale, mode='bilinear', align_corners=False)

        l1_post = torch.cat([post, img_ref], dim=1)
        l1_post_offset = self.lrelu(self.L1_post_offset1(l1_post))
        l1_post_offset = self.lrelu(self.L1_post_offset2(torch.cat([l1_post_offset, l2_post_offset * 2], dim=1)))
        l1_post_offset = self.lrelu(self.L1_post_offset3(l1_post_offset))
        l1_post_max = img_ref - post
        l1_post_max = self.lrelu(self.L1_post_max1(l1_post_max))
        l1_post_max = self.lrelu(self.L1_post_max2(l1_post_max))
        l1_post_dcn = self.L1_post_dcn(post, l1_post_offset)
        l1_post_fuse = self.lrelu(self.L1_post_fuse(torch.cat([l1_post_max, l1_post_dcn], dim=1)))
        l1_post_fuse = self.lrelu(self.L1_post_fuseL2(torch.cat([l1_post_fuse, l2_post_fuse], dim=1)))

        return l1_pre_fuse, l1_post_fuse


class ImageVSR(nn.Module):
    def __init__(self, nf=64, num_res=5):
        super(ImageVSR, self).__init__()
        self.scale = opt.scale
        self.fea_ext = FeaExt(nf)
        self.multi_PC = MultiPCAligned(nf)

        self.ref_expand = Conv2d(nf, nf * 2, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_expand = Conv2d(nf, nf * 2, (3, 3), stride=(1, 1), padding=(1, 1))
        self.post_expand = Conv2d(nf, nf * 2, (3, 3), stride=(1, 1), padding=(1, 1))

        self.feat_ref1 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_ref2 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_ref3 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_pre1 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_pre2 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_post1 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])
        self.feat_post2 = Sequential(*[Res_Block2D(nf * 2) for _ in range(num_res)])

        self.upconv1 = Conv2d(nf * 2, nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2 = Conv2d(nf, nf * 4, (3, 3), stride=(1, 1), padding=(1, 1))

        self.hr = Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.last = Conv2d(nf, 3, (3, 3), stride=(1, 1), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inputs):
        n, _, d, size_h, size_w = inputs.shape
        idx = d // 2
        img_ref = inputs[:, :, idx, :, :]
        img_ref_bic = F.interpolate(img_ref, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # First stage alignment
        img_pre, img_ref, img_post = self.fea_ext(inputs, idx)

        #  Second stage alignment
        pre_result, post_result = self.multi_PC(img_pre, img_ref, img_post)

        pre_result = self.lrelu(self.pre_expand(pre_result))
        img_ref = self.lrelu(self.ref_expand(img_ref))
        post_result = self.lrelu(self.post_expand(post_result))

        #  Residual difference fusion
        feat_recons = self.feat_ref1(img_ref)
        pre_result = self.feat_pre1(pre_result)
        x1 = feat_recons - pre_result
        x1 = self.feat_pre2(x1)
        feat_recons = feat_recons + x1
        feat_recons = self.feat_ref2(feat_recons)
        post_result = self.feat_post1(post_result)
        x2 = feat_recons - post_result
        x2 = self.feat_post2(x2)
        feat_recons = feat_recons + x2
        feat_recons = self.feat_ref3(feat_recons)

        # Sub-pixel magnification
        f = self.lrelu(self.upconv1(feat_recons))
        f = F.pixel_shuffle(f, 2)
        f = self.lrelu(self.upconv2(f))
        f = F.pixel_shuffle(f, 2)
        f = self.lrelu(self.hr(f))
        f = self.lrelu(self.last(f))

        outputs = f + img_ref_bic

        return outputs

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, Conv2d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)
        if isinstance(m, Conv3d):
            nn.init.xavier_normal_(m.weight, gain=1.)
            nn.init.zeros_(m.bias)


