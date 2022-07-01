import os
import glob
import random
import torch
import img_preprocess
from option import opt
from torch.utils import data
from torchvision import transforms
from Guassian import Guassian_downsample


def train_process(HR):
    group = []
    idx = opt.num_frames // 2
    group.append(HR[0])
    group.append(HR[idx])
    group.append(HR[1])

    group.append(HR[idx])

    group.append(HR[-2])
    group.append(HR[idx])
    group.append(HR[-1])

    return group


class TrainData(data.Dataset):
    def __init__(self):
        self.train_data = open(opt.train_data, 'rt').read().splitlines()
        self.scale = opt.scale
        self.num_frames = opt.num_frames
        self.trans_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        HR_all = []
        img_path = sorted(glob.glob(os.path.join('../autodl-tmp/data81/sequences', self.train_data[idx], '*.png')))
        if random.random() < 0.25:
            img_trans_args = 100
        else:
            img_trans_args = random.randint(-1, 1)

        start_frames = random.randint(0, 2)
        for i in range(self.num_frames):
            img = img_preprocess.imread(img_path[start_frames + i])
            if img_trans_args == -1 or img_trans_args == 0 or img_trans_args == 1:
                img = img_preprocess.img_trans(img, img_trans_args)
            img_crop = img_preprocess.modcropHR(img)
            img_nor = img_preprocess.img_normal(img_crop)
            HR = self.trans_tensor(img_nor).float()
            HR_all.append(HR)
        HR_end = train_process(HR_all)
        HR_end = torch.stack(HR_end, dim=1)
        LR = Guassian_downsample(HR_end, self.scale)
        GT = HR_end[:, 1, :, :]

        return LR, GT

    def __len__(self):
        return len(self.train_data)
