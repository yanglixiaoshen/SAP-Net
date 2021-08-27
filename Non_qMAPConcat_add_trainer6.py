
# train: ehance.epoch444. batch_size = 8 P=15 lr=1e-4 validate: ehance.epoch xxx, 418; iqa.epoch64, 60
from argparse import ArgumentParser
import os
import numpy as np
import random
from scipy import stats
import yaml
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from data.yl360IQAData import IQADataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from tensorboardX import SummaryWriter
import datetime
from option import args
import utility
from model.DenseWTUnet import BSR
#from model import Model
from loss import Loss
import logging
import shutil
import time
import math
import os
import matplotlib.pyplot as plt
from importlib import import_module
from data.yl360IQAData import *
from model.Non_qMAPConcat_add_IQA import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def metricIQA(y, y_pred):
    gt = np.reshape(y, (-1,))

    pr = np.reshape(y_pred, (-1,))

    srocc = stats.spearmanr(gt, pr)[0]
    krocc = stats.stats.kendalltau(gt, pr)[0]
    plcc = stats.pearsonr(gt, pr)[0]
    rmse = np.sqrt(((gt-pr) ** 2).mean())
    mae = np.abs((gt-pr)).mean()
    #outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

    return srocc, krocc, plcc, rmse, mae

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

def validate(mw_model, model, val_loader):

    mw_model.eval()
    model.eval()
    scores, gt_scores = [], []
    for num, val_batch in enumerate(val_loader):
        im_mw, imp_iwt, gt_iwt, im_dmos = val_batch
        print(im_mw.size())
        #print(imp_dwt)
        pre_iwt = mw_model(im_mw)

        pre_iwt = [LocalNormalization(pre_iwt[i][0].detach().cpu().numpy()) for i in range(pre_iwt.size(0))]
        pre_iwt = torch.stack(pre_iwt).cuda()

        pre_score = model(imp_iwt, pre_iwt - imp_iwt)
        scores.append(pre_score.squeeze(0).detach().cpu().numpy())
        gt_scores.append(im_dmos.squeeze(0).cpu().numpy())
    scores = np.array(scores)
    gt_scores = np.array(gt_scores)
    print(scores, gt_scores)
    srocc, krocc, plcc, rmse, mae = metricIQA(scores, gt_scores)


    return srocc, krocc, plcc, rmse, mae


def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir,
        disable_gpu=False):
    #print(config)
    if config['test_ratio'] is not None:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    module = import_module('model.' + 'MWCNN')
    mw_model = module.make_model(args).to('cuda')

    model = Model(args).to('cuda')

    writer = SummaryWriter(log_dir=log_dir)


    if os.path.exists(os.path.join(args.log_dir_MW, "state.pkl.epoch444")):
        mw_model.load_state_dict(torch.load(os.path.join(args.log_dir_MW, "state.pkl.epoch444")), strict=False) #
        logger.info("Successfully loaded pretrained Epoch_MW_model.")
    else:
        mw_model.load_state_dict(torch.load(os.path.join(args.log_dir_MW, "state.pkl.epoch418")), strict=False)  #
        logger.info("Successfully loaded pretrained newly saved MW_model.")


    if os.path.exists(os.path.join(args.log_dir_IQA7, "state.pkl")):
        model.load_state_dict(torch.load(os.path.join(args.log_dir_IQA7, "state.pkl")), strict=False) #
        logger.info("Successfully loaded pretrained IQA_model.")


    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if os.path.exists(os.path.join(args.log_dir_IQA7, "optimizer_state.pkl")):
        optimizer.load_state_dict(torch.load(os.path.join(args.log_dir_IQA7, "optimizer_state.pkl")))
        logger.info("Successfully loaded optimizer IQA_parameters.")

    loss_avg = Loss(args)
    iter = 0
    for epoch in range(epochs)[1:]:
        epoch_loss = []

        for batch_num, (im_mw, imp_iwt, gt_iwt, im_dmos) in enumerate(train_loader):
            iter += 1
            mw_model.eval()
            model.train()
            optimizer.zero_grad()
            pre_iwt = mw_model(im_mw)

            pre_iwt = [LocalNormalization(pre_iwt[i][0].detach().cpu().numpy()) for i in range(train_batch_size)]
            pre_iwt = torch.stack(pre_iwt).cuda()

            error_map = pre_iwt - imp_iwt
            #print(imp_iwt, error_map)
            pre_score = model(imp_iwt, error_map)

            loss_batch = loss_avg(pre_score, im_dmos)

            plt.imsave(os.path.join(args.log_dir_IQA7, 'hr.jpg'), gt_iwt.detach().cpu().numpy()[0][0])
            plt.imsave(os.path.join(args.log_dir_IQA7, 'sr.jpg'), pre_iwt.detach().cpu().numpy()[0][0])
            plt.imsave(os.path.join(args.log_dir_IQA7, 'lr.jpg'), imp_iwt.detach().cpu().numpy()[0][0])

            loss_batch.backward()
            optimizer.step()

            torch.save(model.state_dict(), os.path.join(args.log_dir_IQA7, "state.pkl"))
            torch.save(optimizer.state_dict(), os.path.join(args.log_dir_IQA7, "optimizer_state.pkl"))

            logger.info("[EPOCH{}:ITER{}] <LOSS>={:.4}".format(epoch, iter, loss_batch.item()))
            writer.add_scalar('Train/Iter/Loss', loss_batch.item(), iter)

            epoch_loss.append(loss_batch.item())

        epoch_loss_log = np.mean(epoch_loss)

        writer.add_scalar('Train/Epoch/Loss', epoch_loss_log, epoch)
        with torch.no_grad():
            mw_model.eval()
            model.eval()
            srocc, krocc, plcc, rmse, mae = validate(mw_model, model, val_loader)

            logger.info("Validation Results - Epoch: {} <PLCC>: {:.4f} <SROCC>: {:.4f} <KROCC>: {:.4f}  <RMSE>: {:.6f} <MAE>: {:.6f}"
                .format(epoch, plcc, srocc, krocc, rmse, mae))

            writer.add_scalar("validation/SROCC", srocc, epoch)
            writer.add_scalar("validation/KROCC", krocc, epoch)
            writer.add_scalar("validation/PLCC", plcc, epoch)
            writer.add_scalar("validation/RMSE", rmse, epoch)
            writer.add_scalar("validation/MAE", mae, epoch)


        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(args.log_dir_IQA7, "state.pkl.epoch{}".format(epoch)))
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

    ensure_dir(args.log_dir_IQA7)

    log_dir = '{}/{}'.format(args.log_dir_IQA7, 'tf')

    ensure_dir(log_dir)

    shutil.copy2(__file__, os.path.join(args.log_dir_IQA7, "script.py"))  # copy2:复制文件和状态到后一个文件  print(__file__)   打印所执行文件当前的位置路径
    # shutil.copy2(BBBRNNModel.__file__, os.path.join(args.log_dir, "model.py"))

    # arguments = copy.deepcopy(locals())
    logger = logging.getLogger("train-IQA")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir_IQA7, "log_train.txt"))
    logger.addHandler(fh)

    run(args.batch_size_iqa, args.epochs, args.lr_iqa, args.weight_decay, config, args.exp_id,
        log_dir, args.disable_gpu)
