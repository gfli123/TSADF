import glob
import os
import cv2
import torch
import img_preprocess
from loss_functions import PSNR, SSIM
from option import opt
from torchvision import transforms
from Guassian import Guassian_downsample
from torch.autograd import Variable
from build_models import ImageVSR


def test(model, use_gpu):
    test_datasets = open(opt.test_data, 'rt').read().splitlines()
    for data_test in test_datasets:
        test_list = os.listdir(data_test)
        psnr_datasets_all = 0
        ssim_datasets_all = 0
        for test_name in test_list:
            with open('./results/test_results.txt', 'a+') as f:
                f.write('--------------------{}--------------------'.format(test_name)+ '\n')
            print('--------------------{}--------------------'.format(test_name))
            inList = sorted(glob.glob(os.path.join(data_test, test_name, '*.png')))
            psnr_all = 0
            ssim_all = 0
            trans_tensor = transforms.ToTensor()
            for i in range(0, len(inList)):
                inputs_all = []
                if i == 0:
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 2])
                elif i == 1:
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 2])
                elif i == len(inList) - 2:
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 1])
                elif i == len(inList) - 1:
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i])
                else:
                    inputs_all.append(inList[i - 2])
                    inputs_all.append(inList[i - 1])
                    inputs_all.append(inList[i])
                    inputs_all.append(inList[i + 1])
                    inputs_all.append(inList[i + 2])
                HR_all = []
                for j in range(opt.num_frames):
                    img = img_preprocess.imread(inputs_all[j])
                    img_crop = img_preprocess.modcrop(img, opt.scale * 4)
                    img_nor = img_preprocess.img_normal(img_crop)
                    HR = trans_tensor(img_nor).float()
                    HR_all.append(HR)
                HR_end = train_process(HR_all)
                HR_end = torch.stack(HR_end, dim=1)

                LR = Guassian_downsample(HR_end, opt.scale)
                
                LR = LR.unsqueeze(0)
                labels = HR_end[:, 1, :, :]
                labels = labels.unsqueeze(0)
                if use_gpu:
                    LR = Variable(LR).cuda()
                    labels = Variable(labels).cuda()

                with torch.no_grad():
                    outputs = model(LR)

                # Y channels calculation
                outputs = outputs.squeeze(0)
                outputs = outputs.permute(1, 2, 0).cpu().numpy()
                save_img(outputs, data_test, test_name, False, i)
                outputs = torch.from_numpy(img_preprocess.bgr2ycbcr(outputs)).unsqueeze(0).unsqueeze(0)
                outputs = outputs.cuda()
                labels = labels.squeeze(0)
                labels = labels.permute(1, 2, 0).cpu().numpy()
                labels = torch.from_numpy(img_preprocess.bgr2ycbcr(labels)).unsqueeze(0).unsqueeze(0)
                labels = labels.cuda()

                psnr = PSNR(labels, outputs)
                ssim = SSIM(labels, outputs)
                psnr_all += psnr
                ssim_all += ssim
                print('psnr = {:.6f}, ssim = {:.6f},'.format(psnr, ssim))
                with open('./results/test_results.txt', 'a+') as f:
                    f.write('psnr = {:.6f}, ssim = {:.6f},'.format(psnr, ssim) + '\n')
            psnr_avg = psnr_all / len(inList)
            ssim_avg = ssim_all / len(inList)
            print('==> Average PSNR = {:.6f}'.format(psnr_avg))
            print('==> Average SSIM = {:.6f}'.format(ssim_avg))
            with open('./results/test_results.txt', 'a+') as f:
                f.write('==> Average PSNR_Avg = {:.6f}'.format(psnr_avg) + '\n')
                f.write('==> Average SSIM_Avg = {:.6f}'.format(ssim_avg) + '\n')
            psnr_datasets_all += psnr_avg
            ssim_datasets_all += ssim_avg
        psnr_datasets_avg = psnr_datasets_all / len(test_list)
        ssim_datasets_avg = ssim_datasets_all / len(test_list)
        print('==> Average PSNR = {:.6f}'.format(psnr_datasets_avg))
        print('==> Average SSIM = {:.6f}'.format(ssim_datasets_avg))
        with open('./results/test_results.txt', 'a+') as f:
            f.write('==> {} Average PSNR_Avg = {:.6f}'.format(data_test, psnr_datasets_avg) + '\n')
            f.write('==> {} Average SSIM_Avg = {:.6f}'.format(data_test, ssim_datasets_avg) + '\n')

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


def save_img(prediction, data_test, test_name, att, num):
    '''

    :param prediction:  img
    :param data_test:   path1
    :param test_name:   path2
    :param att: True img [0, 255], False [0, 1]
    :return:
    '''

    if att == True:
        save_dir = os.path.join(opt.image_out, data_test, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        save_dir = os.path.join(opt.image_out, data_test, test_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_dir = os.path.join(save_dir, '{:03}'.format(num) + '.png')
        cv2.imwrite(image_dir, prediction*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    model = ImageVSR()

    checkpoints = torch.load(opt.pretrain)
    model.load_state_dict(checkpoints['net'])

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    test(model, use_gpu)


