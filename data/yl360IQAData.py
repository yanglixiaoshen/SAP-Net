
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
import random
import model.common as common
from option import args

def default_loader(path):
    return Image.open(path).convert('L')  # RGB-->Gray


def LocalNormalization(patch, P=15, Q=15, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln

def LocalNormalization_test(patch, P=15, Q=15, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0).unsqueeze(0)
    return patch_ln

def NonOverlappingCropPatches(im, patch_size=256, stride=256):
    # 8000, 4000
    w, h = im.size
    patches = ()
    patches_mw = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches_mw = patches_mw + (patch.unsqueeze(0),)
            patch = LocalNormalization_test(patch[0].numpy())
            patches = patches + (patch,)  # great !!!
    return patches_mw, patches

# IQA 去掉顶部的两个stride
def NonOverlappingCropPatches_1(im, patch_size=256, stride=256):
    # 8000, 4000
    w, h = im.size
    patches = ()
    patches_mw = ()
    for i in range(stride*3, h - stride*2, stride):
        for j in range(stride*2, w - stride*2, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches_mw = patches_mw + (patch.unsqueeze(0),)
            patch = LocalNormalization_test(patch[0].numpy())
            patches = patches + (patch,)  # great !!!
    return patches_mw, patches


# For training Enhance model
def NonOverlappingCropPatches_random(im, gt, patch_size=32, stride=32):
    w, h = im.size # 8000, 4000
    rnd_h = random.randint(0, max(0, h - patch_size))
    rnd_w = random.randint(0, max(0, w - patch_size))

    im_crop = im.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
    im_crop = np.asarray(im_crop)

    gt_crop = gt.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
    gt_crop = np.asarray(gt_crop)

    #im_crop = torch.from_numpy()
    return im_crop, gt_crop
# For training IQA model
def NonOverlappingCropPatchesIQA_random(im, gt, patch_size=32, stride=32):
    w, h = im.size # 8000, 4000
    rnd_h = random.randint(0, max(0, h - patch_size))
    rnd_w = random.randint(0, max(0, w - patch_size))

    im_mw = to_tensor(im.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size)))

    im_crop = im.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
    im_crop = LocalNormalization(to_tensor(im_crop)[0].numpy())

    gt_crop = gt.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
    gt_crop = LocalNormalization(to_tensor(gt_crop)[0].numpy())

    #im_crop = torch.from_numpy()
    return im_mw, im_crop, gt_crop

class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.imp_num = conf['imp_num']
        self.loader = loader
        self.imrefID = conf['yl360Dataset1']['refimpID_pth']
        self.im_dir = conf['yl360Dataset1']['img_ref_IMG_pth']
        self.im_dmos = conf['yl360Dataset1']['impDMOS_reg']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        self.bz = conf['batch_size']
        self.dwt = common.DWT()
        self.indexData = np.arange(self.imp_num)
        self.dmos = np.loadtxt(self.im_dmos)

        if os.path.exists(os.path.join(args.log_dir_MW, "train_test_randList.txt")):
            self.indexData = np.loadtxt(os.path.join(args.log_dir_MW, "train_test_randList.txt"))
        else:
            np.random.shuffle(self.indexData)
            np.savetxt(os.path.join(args.log_dir_MW, "train_test_randList.txt"), self.indexData)

        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = self.indexData[: int(train_ratio*self.imp_num)]
        valindex = self.indexData[int(train_ratio*self.imp_num): int((1 - test_ratio) * self.imp_num)+1]
        testindex = self.indexData[int((1 - test_ratio) * self.imp_num): ]

        if status == 'train':
            self.index = trainindex
            np.savetxt(os.path.join(args.log_dir_IQA, 'train_score.txt'), self.index)
            print(len(self.index))
            print("# Train Images: {}".format(len(self.index)))
            print(trainindex)
        if status == 'test':
            self.index = testindex
            np.savetxt(os.path.join(args.log_dir_IQA, 'test_score.txt'), self.index)
            print(len(self.index))
            print("# Test Images:  {}".format(len(self.index)))
            print(testindex)
        if status == 'val':
            self.index = valindex
            np.savetxt(os.path.join(args.log_dir_IQA, 'validate_score.txt'), self.index)
            print("# Val Images: {}".format(len(self.index)))


    def __len__(self):
        return len(self.index) # 960 / 5 =192

    def __getitem__(self, idx):
        imp_id = self.index[idx]
        #print(imp_id)
        #print(np.loadtxt(self.imrefID, dtype=str).shape)
        im_dmos = self.dmos[int(imp_id)]
        gt_name = np.loadtxt(self.imrefID, dtype=str)[int(imp_id), 0]
        imp_name = np.loadtxt(self.imrefID, dtype=str)[int(imp_id), 1]
        #print(gt_name)
        gt = self.loader('.'.join([os.path.join(self.im_dir, gt_name), 'jpg']))
        imp = self.loader('.'.join([os.path.join(self.im_dir, imp_name), 'jpg']))
        #gt_crop_np = NonOverlappingCropPatches_random(gt, self.patch_size, self.stride)
        im_mw, imp_crop_tor, gt_crop_tor = NonOverlappingCropPatchesIQA_random(imp, gt, self.patch_size, self.stride)
        #print(im_dmos)
        return im_mw.float().cuda(), imp_crop_tor.cuda(), gt_crop_tor.cuda(), torch.Tensor([im_dmos]).float().cuda()
        # 返回一个img的一个patch, 归一化patch, GT归一化patch以及其分数标签。
