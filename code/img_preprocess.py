import cv2
import numpy as np
from option import opt

def img_normal(img_inputs):
    img  = np.float32(img_inputs / 255.)
    return img

def imread(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def img_trans(img_inputs, args):
    img = cv2.flip(img_inputs, args)
    return img


def modcropHR(img_inputs):
    img = np.copy(img_inputs)
    if img.ndim == 3:
        H_r = opt.crop_size
        W_r = opt.crop_size
        img = img[:H_r, :W_r, :]
    return img


def modcrop(img_inputs, scales):
    img = np.copy(img_inputs)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r = H % scales
        W_r = W % scales
        img = img[:H - H_r, :W - W_r, :]

    return img


def bgr2ycbcr(img_inputs, only_y=True):
    # img_inputs: BGR channels
    '''
    only_y: only return Y channel
    img_Input:
        [0, 255]
    '''
    # convert
    if only_y:
        img_trans = np.dot(img_inputs, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        img_trans = np.matmul(img_inputs, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    img_outputs = np.float32(img_trans)
    return img_outputs

