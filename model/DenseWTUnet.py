'''
@Description:
@Author: fangn
@Github:
@Date: 2019-11-18 10:23:32
@LastEditors: fangn
@LastEditTime: 2019-11-19 10:23:12
'''
from model import common
import torch
import torch.nn as nn
import scipy.io as sio
from collections import OrderedDict

# 创建模型
def make_model(args, parent=False):
    return DenseWTUnet(growth_rate=32, block_config=(6, 12), num_init_features=4,
                 bn_size=4, drop_rate=0)

def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
    mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
    mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()

# BSR 模型
class BSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3  # 卷积核大小
        self.scale_idx = 0

        act = nn.ReLU(True)  # 激活函数

        self.DWT = common.DWT()  # 二维离散小波
        self.IWT = common.IWT()  # 逆向的二维离散小波

        n = 3
        # downsample的第一层，维度变化4->16
        m_head = [common.BBlock(conv, 4, 160, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(common.BBlock(conv, 160, 160, 3, act=act))

        # downsample的第二层，维度变化640->256（默认的feature map == 64）
        d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        # downsample的第三层，并与upsample进行连接，也是upsample的第三层
        # 维度变化1024->256，256->1024
        pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]
        for _ in range(n * 2):
            pro_l3.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        pro_l3.append(
            common.BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))

        # upsample的第二层，维度变化256->640
        i_l2 = []
        for _ in range(n):
            i_l2.append(
                common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        i_l2.append(common.BBlock(conv, n_feats * 4, 640, 3, act=act))

        # upsample的第一层，维度变化160->4
        i_l1 = []
        for _ in range(n):
            i_l1.append((common.BBlock(conv, 160, 160, 3, act=act)))
        m_tail = [conv(160, 4, 3)]

        # downsample的第一层
        self.head = nn.Sequential(*m_head)
        self.d_l1 = nn.Sequential(*d_l1)
        # downsample的第二层
        self.d_l2 = nn.Sequential(*d_l2)
        # 第三层连接层
        self.pro_l3 = nn.Sequential(*pro_l3)
        # upsample的第二层
        self.i_l2 = nn.Sequential(*i_l2)
        # upsample的第一层
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # downsample的第一层
        x1 = self.d_l1(self.head(self.DWT(x))) # 160
        # downsample的第二层
        x2 = self.d_l2(self.DWT(x1)) # 256
        # upsample的第三层，并且使用了short cut的结构，将对应的downsample的第二层
        # 加到upsample对应的层上。
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        # upsample的第二层
        x_ = self.IWT(self.i_l2(x_)) + x1
        # upsample的第一层
        x = self.IWT(self.tail(self.i_l1(x_))) + x
        # x = self.add_mean(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

class DenseWTUnet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12), num_init_features=4,
                 bn_size=4, drop_rate=0):
        super(DenseWTUnet, self).__init__()

        self.DWT = common.DWT()  # 二维离散小波
        self.IWT = common.IWT()  # 逆向的二维离散小波

        # out: b, 4, 128, 128
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(4, num_init_features, kernel_size=1, stride=1, padding=0, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True))
        ]))
        num_layers1 = block_config[0]
        num_layers2 = block_config[1]
        # print(num_layers2)
        # out: b, 196, 128, 128
        self.block1 = common._DenseBlock(num_layers1, num_init_features, bn_size, growth_rate, drop_rate)
        num_features1 = num_init_features + growth_rate * num_layers1  # 196
        # out: b, 1168, 64, 64
        self.block2 = common._DenseBlock(num_layers2, num_features1 * 4, bn_size, growth_rate, drop_rate)
        num_features2 = num_features1 *4 + growth_rate * num_layers2
        # out: b, 256, 64, 64
        self.features1 = nn.Sequential(OrderedDict([
            ("conv11", nn.Conv2d(num_features2, 256, kernel_size=1, stride=1, padding=0, bias=False)),
            ("norm11", nn.BatchNorm2d(256)),
            ("relu11", nn.ReLU(inplace=True))
        ]))
        self.block3 = common._DenseBlock(num_layers2, 256, bn_size, growth_rate, drop_rate)
        #num_features3 = 256 + growth_rate * num_layers2 # 640
        # out: b, 24, 128, 128
        self.features2 = nn.Sequential(OrderedDict([
            ("conv22", nn.Conv2d(160, 24, kernel_size=1, stride=1, padding=0, bias=False)),
            ("norm22", nn.BatchNorm2d(24)),
            ("relu22", nn.ReLU(inplace=True))
        ]))
        # out: b, 256, 64, 64
        self.block4 = common._DenseBlock(num_layers1, num_features1 + 24, bn_size, growth_rate, drop_rate)
        num_features3 = 220 + growth_rate * num_layers1 # 412

        self.features3 = nn.Sequential(OrderedDict([
            ("conv33", nn.Conv2d(num_features3, 4, kernel_size=1, stride=1, padding=0, bias=False)),
            ("norm33", nn.BatchNorm2d(4)),
            ("relu33", nn.ReLU(inplace=True))
        ]))

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #print(x.size())
        x1 = self.features(x) # x1: (5, 4, 128, 128)
        x21 = self.block1(x1)
        x2 = self.block3(self.features1(self.block2(self.DWT(x21))))
        x3 = self.features2(self.IWT(x2))
        #print(x2.size(), x3.size())
        x4 = torch.cat((x21, x3), 1)
        x4 = self.block4(x4)
        x5 = self.features3(x4) + x

        x_ehan = self.IWT(x5)

        return x5, x_ehan


