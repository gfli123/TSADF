import datetime
import os
import time
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from option import opt
from datasets import TrainData
from torch.utils.data import DataLoader
from build_models import ImageVSR, initialize_weights
from torch.autograd import Variable

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
def main():
    torch.manual_seed(opt.seed)
    train_data = TrainData()
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.threads, pin_memory=False, drop_last=False)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

    model = ImageVSR()
    model.apply(initialize_weights)
    print("Model_add size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) * 4 / 1048576))
    args = sum(p.numel() for p in model.parameters()) / 1000000
    print('args=', args)
    criterion_mse = nn.MSELoss()

    if use_gpu:
        model = model.cuda()
        criterion_mse = criterion_mse.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=opt.gamma)

    start_epoch = 0

    for epochs in range(start_epoch, opt.num_epochs):
        loss_all = []
        for steps, data in enumerate(train_dataloader):
            start_time = time.time()
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

            outputs = model(inputs)
            loss_mse = criterion_mse(labels, outputs)
            loss_all.append(loss_mse)
            optimizer.zero_grad()
            loss_mse.backward()
            optimizer.step()
            end_time = time.time()
            cost_time = end_time - start_time
            if steps % 30 == 0:
                print('===> Epochs[{}]({}/{}) || Time = {:.3f}s'.format(epochs, steps+1, len(train_dataloader), cost_time),
                      'loss_mse = {:.8f}'.format(loss_mse))
        scheduler.step()
        if epochs % 10 == 0:
            save_models(model, optimizer, epochs)


def save_models(model, optimizer, epochs):
    save_model_path = os.path.join(opt.save_model_path, systime)
    if not os.path.exists(save_model_path):
        os.makedirs(os.path.join(save_model_path))
    save_name = 'X' + str(opt.scale) + '_epoch_{}.pth'.format(epochs)
    checkpoint = {"net": model.state_dict(), 'optimizer': optimizer.state_dict(), "epoch": epochs}
    torch.save(checkpoint, os.path.join(save_model_path, save_name))
    print('Checkpoints save to {}'.format(save_model_path))


if __name__ == '__main__':
    main()
