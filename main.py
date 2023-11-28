import datetime
import os
import time
import torch
import logging
from torch import nn, optim
from torch.optim import lr_scheduler
from logger import load_logger
from option import opt
from datasets import Train_Vimeo
from torch.utils.data import DataLoader
from build_models import VSR, initialize_weights
from torch.autograd import Variable

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
def main():
    torch.manual_seed(opt.seed)

    if opt.train_datasets == 'Vimeo-90K':
        train_data = Train_Vimeo()
    elif opt.train_datasets == 'MM522':
        train_data = Train_MM522()
    else:
        raise Exception('No training set, please choose a training set')

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.threads, pin_memory=False, drop_last=False)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

    model = VSR()
    criterion = nn.L1Loss()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    if not os.path.exists('./results'):
        os.makedirs('./results')

    logger = load_logger(logger_name='VSR')

    logger.error("Model_add size: {:.5f}MB".format(sum(p.numel() for p in model.parameters()) * 4 / 1048576))
    logger.error('Params = {:.6f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    start_epoch = 0

    resume = False
    if resume:
        path_checkpoint = "./checkpoints/X4_epoch_5.pth"
        checkpoints = torch.load(path_checkpoint)
        model.load_state_dict(checkpoints['net'])
        start_epoch = checkpoints['epoch']

    for epochs in range(start_epoch + 1, opt.num_epochs + 1):
        loss_all = []
        logger.debug('第%d个epoch的学习率：%f' % (epochs, optimizer.param_groups[0]['lr']))

        for steps, data in enumerate(train_dataloader):
            start_time = time.time()
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

            outputs = model(inputs)
            loss_mse = criterion(labels, outputs)

            loss_all.append(loss_mse.item())

            optimizer.zero_grad()
            loss_mse.backward()

            nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            end_time = time.time()
            cost_time = end_time - start_time
            logger.debug('===> Epochs[{}]({}/{}) || Time = {:.3f}s, loss_mse = {:.8f}'.format(epochs, steps + 1, len(train_dataloader), cost_time, loss_mse))

        scheduler.step()

        logger.warning('Epochs[{}] || Loss_MSE = {:.6f}'.format(epochs, func_sum(loss_all)))

        if epochs % 5 == 0:
            save_models(model, epochs, logger)


def save_models(model, epochs, logger):
    save_model_path = os.path.join(opt.save_model_path, systime)
    if not os.path.exists(save_model_path):
        os.makedirs(os.path.join(save_model_path))
    save_name = 'X' + str(opt.scale) + '_epoch_{}.pth'.format(epochs)
    checkpoint = {"net": model.state_dict(), "epoch": epochs}
    torch.save(checkpoint, os.path.join(save_model_path, save_name))
    logger.debug('Checkpoints save to {}'.format(save_model_path))


def func_sum(loss):
    outputs = sum(loss)/len(loss)
    return outputs


if __name__ == '__main__':
    main()
