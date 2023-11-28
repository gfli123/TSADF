import glob
import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from loss_functions import PSNR, SSIM
from option import opt
from torchvision import transforms
from Guassian import Guassian_downsample
from torch.autograd import Variable
from build_models import VSR
from logger import load_logger
from img_preprocess import imread, img_trans, modcrop, img_normal, bgr2ycbcr, modcrop_size


def test_Vid4(model, epochs, use_gpu, logger, device_ids=[0, 1]):
    test_datasets = open(opt.test_data, 'rt').read().splitlines()
    for data_test in test_datasets:
        test_list = os.listdir(data_test)
        psnr_datasets_all = 0
        ssim_datasets_all = 0
        time_datasets_all = 0
        for test_name in test_list:
            inList = sorted(glob.glob(os.path.join(data_test, test_name, '*.png')))
            psnr_all = 0
            ssim_all = 0
            trans_tensor = transforms.ToTensor()
            time_all = 0
            logger.info('----------{}----------'.format(test_name))
            for i in range(0, len(inList)):
                inputs_all = []
                if i == 0:
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[1])
                    inputs_all.append(inList[2])
                    inputs_all.append(inList[3])
                elif i == 1:
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[1])
                    inputs_all.append(inList[2])
                    inputs_all.append(inList[3])
                    inputs_all.append(inList[4])
                elif i == 2:
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[0])
                    inputs_all.append(inList[1])
                    inputs_all.append(inList[2])
                    inputs_all.append(inList[3])
                    inputs_all.append(inList[4])
                    inputs_all.append(inList[5])
                elif i == len(inList) - 1:
                    inputs_all.append(inList[i - 3])
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                elif i == len(inList) - 2:
                    inputs_all.append(inList[i - 3])
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 1])
                elif i == len(inList) - 3:
                    inputs_all.append(inList[i - 3])
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 2])
                    inputs_all.append(inList[i + 2])
                else:
                    inputs_all.append(inList[i - 3])
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 2])
                    inputs_all.append(inList[i + 3])
                HR_all = []
                for j in range(opt.num_frames):
                    img = imread(inputs_all[j])
                    img = modcrop_size(img, opt.scale ** 2)
                    HR_all.append(img)
                HR_all = [trans_tensor(HR) for HR in HR_all]
                HR_all = torch.stack(HR_all, dim=1)
                labels = HR_all[:, opt.num_frames // 2, :, :]

                LR = Guassian_downsample(HR_all, opt.scale)
                LR = LR.permute(1, 0, 2, 3).unsqueeze(0)

                if use_gpu:
                    LR = Variable(LR).cuda(device=device_ids[0])
                    labels = Variable(labels).cuda(device=device_ids[0])
                start_time = time.time()
                with torch.no_grad():
                    prediction = model(LR)
                end_time = time.time()
                cost_time = end_time - start_time
                time_all += cost_time
                prediction = prediction.squeeze(0)

                # Y channels calculation
                prediction = prediction.permute(1, 2, 0).cpu().numpy()
                save_img(prediction, data_test, test_name, False, i)
                prediction = torch.from_numpy(bgr2ycbcr(prediction)).unsqueeze(0).unsqueeze(0)
                prediction = prediction.cuda()

                labels = labels.permute(1, 2, 0).cpu().numpy()
                labels = torch.from_numpy(bgr2ycbcr(labels)).unsqueeze(0).unsqueeze(0)
                labels = labels.cuda()

                psnr = PSNR(labels, prediction)
                ssim = SSIM(labels, prediction)
                psnr_all += psnr
                ssim_all += ssim
                logger.info('epochs:{},  psnr = {:.6f}, ssim={:.6f},'.format(epochs, psnr, ssim))

            psnr_avg = psnr_all / len(inList)
            ssim_avg = ssim_all / len(inList)
            time_avg = time_all / len(inList)
            logger.info('Average running time = {:.6f}'.format(time_avg))
            logger.info('Epochs:{}, Average PSNR_Avg = {:.6f}'.format(epochs, psnr_avg))
            logger.info('Epochs:{}, Average SSIM_Avg = {:.6f}'.format(epochs, ssim_avg))

            psnr_datasets_all += psnr_avg
            ssim_datasets_all += ssim_avg
            time_datasets_all += time_avg
        psnr_datasets_avg = psnr_datasets_all / len(test_list)
        ssim_datasets_avg = ssim_datasets_all / len(test_list)
        time_datasets_avg = time_datasets_all / len(test_list)
        logger.error('Epochs:{} {} Average PSNR_Avg = {:.6f}, SSIM_Avg = {:.6f}, Time_avg = {:.6f}'
                     .format(epochs, data_test, psnr_datasets_avg, ssim_datasets_avg, time_datasets_avg))


def save_img(prediction, data_test, test_name, att, num):
    '''

    :param prediction:  img
    :param data_test:   path1
    :param test_name:   path2
    :param att: True img [0, 255], False [0, 1]
    :return:
    '''

    dataset_name = data_test.split('/')[-1]

    if att == True:
        save_dir = os.path.join(opt.image_out, dataset_name, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        save_dir = os.path.join(opt.image_out, dataset_name, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    model = VSR()

    path_checkpoint = "./X4_epoch_100.pth"  # 使用断点续弦时修改
    checkpoints = torch.load(path_checkpoint)
    model.load_state_dict(checkpoints['net'])
    epochs = checkpoints['epoch']
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    test_datasets = opt.test_datasets
    if not os.path.exists('./results'):
        os.makedirs('./results')
    logger = load_logger(logger_name='VSR')
    test_Vid4(model, epochs, use_gpu, logger)



