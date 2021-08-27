



import numpy as np
import random
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from data.yl360Dataset import IQADataset
from tensorboardX import SummaryWriter
from option import args
from loss import Loss
import logging
import shutil
import math
import os
import matplotlib.pyplot as plt
from importlib import import_module
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def metricOnBatch(output):
    psnr_batch = []
    y_pred, y = output
    _y_pred = y_pred.detach().cpu().numpy().reshape((y_pred.size(0)*y_pred.size(1), y_pred.size(2), y_pred.size(3)))
    _y      = y.detach().cpu().numpy().reshape((y.size(0)*y.size(1), y.size(2), y.size(3)))

    psnr_batch += [psnr(_y[i], _y_pred[i]) for i in range(y_pred.size(0) * y_pred.size(1))]
    #print(_y[0], _y_pred[0])
    # plt.imsave('/home/yl/logger_enhance/hr.jpg', _y[0])
    # plt.imsave('/home/yl/logger_enhance/sr.jpg', _y_pred[0])
    psnr_fin = np.mean(psnr_batch)
    return psnr_fin


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader

def validate(model, val_loader):
    model.eval()
    psnr_val = []
    for num, val_batch in enumerate(val_loader):
        imp_dwt, gt_dwt, imp_iwt, gt_iwt = val_batch
        #print(imp_dwt)
        pre_iwt = model(imp_iwt)
        psnr_val += [metricOnBatch((pre_iwt, gt_iwt))]
    return np.mean(psnr_val)


def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir,
        disable_gpu=False):
    #print(config)
    if config['test_ratio'] is not None:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    module = import_module('model.' + 'MWCNN_NonDWT')
    model = module.make_model(args).to('cuda')
    writer = SummaryWriter(log_dir=log_dir)
    # model = model.to(device)
    # print(model)

    if os.path.exists(os.path.join(args.log_dir_NonMW1, "state.pkl")):
        model.load_state_dict(torch.load(os.path.join(args.log_dir_NonMW1, "state.pkl")),  strict=False) #
        logger.info("Successfully loaded pretrained MW_model.")

    # if multi_gpu and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if os.path.exists(os.path.join(args.log_dir_NonMW1, "optimizer_state.pkl")):
        optimizer.load_state_dict(torch.load(os.path.join(args.log_dir_NonMW1, "optimizer_state.pkl")))
        logger.info("Successfully loaded optimizer MW_parameters.")

    loss_avg = Loss(args)
    iter = 0
    for epoch in range(epochs)[1:]:
        epoch_loss = []
        epoch_psnr = []
        for batch_num, (imp_dwt, gt_dwt, imp_iwt, gt_iwt) in enumerate(train_loader):
            iter += 1
            model.train()
            optimizer.zero_grad()
            pre_iwt = model(imp_iwt)
            loss_batch = loss_avg(pre_iwt, gt_iwt)
            outputForMetric = (pre_iwt, gt_iwt)
            plt.imsave('/home/yl/logger_enhance_MW/hr.jpg', gt_iwt.detach().cpu().numpy()[0][0])
            plt.imsave('/home/yl/logger_enhance_MW/sr.jpg', pre_iwt.detach().cpu().numpy()[0][0])
            plt.imsave('/home/yl/logger_enhance_MW/lr.jpg', imp_iwt.detach().cpu().numpy()[0][0])
            psnr_batch = metricOnBatch(outputForMetric)
            psnr_ori_batch = metricOnBatch((imp_iwt, gt_iwt))
            #print(imp_iwt, pre_iwt)

            #print(psnr_ori_batch, psnr_batch)
            delta_psnr = psnr_batch - psnr_ori_batch
            loss_batch.backward()
            optimizer.step()

            torch.save(model.state_dict(), os.path.join(args.log_dir_NonMW1, "state.pkl"))
            torch.save(optimizer.state_dict(), os.path.join(args.log_dir_NonMW1, "optimizer_state.pkl"))
            logger.info("[EPOCH{}:ITER{}] <LOSS>={:.4} <TRAIN_PSNR>={:.4} <delta_PSNR>={:.6}".format(epoch, iter, loss_batch.item(), psnr_batch, delta_psnr))

            epoch_loss.append(loss_batch.item())
            epoch_psnr.append(psnr_batch)
            writer.add_scalar('Train/Iter/Loss', loss_batch.item(), iter)
            writer.add_scalar('Train/Iter/PSNR', psnr_batch, iter)
            writer.add_scalar('Train/Iter/delta_PSNR', delta_psnr, iter)
        epoch_loss_log = np.mean(epoch_loss)
        epoch_psnr_log = np.mean(epoch_psnr)

        writer.add_scalar('Train/Epoch/Loss', epoch_loss_log, epoch)
        writer.add_scalar('Train/Epoch/PSNR', epoch_psnr_log, epoch)

        metric_val = validate(model, val_loader)
        writer.add_scalar('Validation/PSNR', metric_val, epoch)
        print('[EPOCH{}:ITER{}] <VAL_PSNR>={:.4}'.format(epoch, iter, metric_val))




        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(args.log_dir_NonMW1, "state.pkl.epoch{}".format(epoch)))
            print('Successfully saved model of EPOCH{}'.format(epoch))


    writer.close()


if __name__ == "__main__":

    torch.set_num_threads(12)
    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ensure_dir(args.log_dir_NonMW1)

    log_dir = '{}/{}'.format(args.log_dir_NonMW1, 'tf')

    ensure_dir(log_dir)


    if not os.path.exists(args.log_dir_NonMW1):
        os.makedirs(args.log_dir_NonMW1)
    shutil.copy2(__file__, os.path.join(args.log_dir_NonMW1, "script.py"))  # copy2:复制文件和状态到后一个文件  print(__file__)   打印所执行文件当前的位置路径


    logger = logging.getLogger("train-ODIEhance_NonMW1")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir_NonMW1, "log_train.txt"))
    logger.addHandler(fh)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        log_dir, args.disable_gpu)
