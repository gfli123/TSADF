import numpy as np
import torch.nn.functional as F
import torch 
'''
   This source file is originally from: https://github.com/junpan19/VSR_TGA
'''

def Guassian_downsample(x, scale=4):
    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        inp[kernlen // 2, kernlen // 2] = 1
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)
    elif scale == 3:
        h = gkern(13, 1.2)
    elif scale == 4:
        h = gkern(13, 1.6)
    else:
        print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(R))
        exit(1)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(C, T, x.size(2), x.size(3))
    return x


