from model import common
import torch
import torch.nn as nn
import scipy.io as sio


# 创建模型
def make_model(args, parent=False):
    return BSR(args)


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
    def __init__(self, args, conv=common.default_conv, conv1 = common.default_conv1):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3  # 卷积核大小
        self.scale_idx = 0

        act = nn.ReLU(True)  # 激活函数

        self.DWT = common.DWT()  # 二维离散小波
        self.IWT = common.IWT()  # 逆向的二维离散小波

        n = 3
        # downsample的第一层，维度变化4->160
        m_head = [common.BBlock(conv, 4, 160, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(common.BBlock(conv, 160, 160, 3, act=act))

        # downsample的第二层，维度变化640->256（默认的feature map == 64）
        d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

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
            i_l1.append((common.BBlock(conv, 160, 160, 3, act = nn.ReLU(False) )))
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
        self.downconv1 = common.BBlock(conv1, 1, 4, 3, act=act)
        self.downconv2 = common.BBlock(conv1, 160, 640, 3, act=act)
        self.downconv3 = common.BBlock(conv1, 256, 1024, 3, act=act)

        self.upconv1 = nn.ConvTranspose2d(1024, 256, 3, 2, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(640, 160, 3, 2, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(4, 1, 3, 2, 1, 1)
    def forward(self, x):
        #x1 = self.d_l1(self.head(self.DWT(x)))
        #         # downsample的第二层
        # x1 = self.d_l1(self.head(self.DWT(x)))
        # x2 = self.d_l2(self.DWT(x1))
        # # upsample的第三层，并且使用了short cut的结构，将对应的downsample的第二层
        # # 加到upsample对应的层上。
        # x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        # # upsample的第二层
        # x_ = self.IWT(self.i_l2(x_)) + x1
        # # upsample的第一层
        # x = self.IWT(self.tail(self.i_l1(x_))) + x
        # # x = self.add_mean(x)
        # return x


        #z = nn.functional.interpolate(y, x.shape[-2:], mode='bilinear')

        #x12 = nn.functional.interpolate(x, (int(x.shape[-2]/2), int(x.shape[-1]/2)), mode='bilinear')
        #x12 = x12.repeat(1, 4, 1, 1)
        #x12 = self.downconv1(x)
        #print(x12)
        # downsample的第一层
        x1 = self.d_l1(self.head(self.downconv1(x)))
        #print(x1)
        # downsample的第二层
        #x11 = nn.functional.interpolate(x1, (int(x1.shape[-2]/2), int(x1.shape[-1]/2)), mode='bilinear')
        #x11 = x11.repeat(1, 640, 1, 1)
        x2 = self.d_l2(self.downconv2(x1))
        x_ = torch.nn.functional.relu(self.upconv1(self.pro_l3(self.downconv3(x2)))) + x2
        x_ = torch.nn.functional.relu(self.upconv2(self.i_l2(x_))) + x1
        #print(x_)
        x = self.upconv3(self.tail(self.i_l1(x_))) + x
        #print(self.i_l1(x_))
        #print(torch.nn.functional.relu(self.upconv3(self.tail(self.i_l1(x_)))))
        return x


        #print(x11) # BSR model 确认是已经写好的NON MW的模型
        #x2 = self.d_l2(x11)
        # upsample的第三层，并且使用了short cut的结构，将对应的downsample的第二层
        # 加到upsample对应的层上。
        #x22 = nn.functional.interpolate(x2, (int(x2.shape[-2]/2), int(x2.shape[-1]/2)), mode='bilinear')
        #x22 = x22.repeat(1, 1024, 1, 1)
        # x22 = self.downconv3(x2)
        # #print(x22)
        # x22 = self.pro_l3(x22)
        # #print(x22)
        # #x22 = nn.functional.interpolate(x22, (int(x22.shape[-2]*2), int(x22.shape[-1]*2)), mode='bilinear')
        # x22 = self.upconv1(x22)
        # x22 = torch.nn.functional.relu(x22)
        # #print(x22)
        #
        # x_ = x22 + x2
        # # upsample的第二层
        # x_ = self.i_l2(x_)
        # #x3 = nn.functional.interpolate(x_, (int(x_.shape[-2] * 2), int(x_.shape[-1] * 2)), mode='bilinear')
        # x3 = self.upconv2(x_)
        # x3 = torch.nn.functional.relu(x3)
        # #print(x3)
        #
        # x3 = x3 + x1
        #
        # x3 = self.tail(self.i_l1(x3))
        # #print(x3)
        # #x3 = nn.functional.interpolate(x3, (int(x3.shape[-2] * 2), int(x3.shape[-1] * 2)), mode='bilinear')
        # x3 = self.upconv3(x3)
        # x3 = torch.nn.functional.relu(x3)
        #
        # # upsample的第一层
        # x = x3 + x
        #print(x3)
        # x = self.add_mean(x)


# from model import common
# import torch
# import torch.nn as nn
# import scipy.io as sio
#
#
# # 创建模型
# def make_model(args, parent=False):
#     return BSR(args)
#
#
# def matrix_init():
#     a = torch.linspace(1, 15, steps=15) - 8
#     a = a.float()
#
#     mat_1 = torch.mul(a, a)
#     mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
#     a = a.view(15, 1)
#     mat_2 = torch.mul(a, a.t())
#     mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
#     mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
#     mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)
#
#     return torch.cat((mat_1, mat_2, mat_3), 1).cuda()
#
#
# # BSR 模型
# class BSR(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(BSR, self).__init__()
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3  # 卷积核大小
#         self.scale_idx = 0
#
#         act = nn.ReLU(True)  # 激活函数
#
#         self.DWT = common.DWT()  # 二维离散小波
#         self.IWT = common.IWT()  # 逆向的二维离散小波
#
#         n = 3
#         # downsample的第一层，维度变化4->16
#         m_head = [common.BBlock(conv, 4, 160, 3, act=act)]
#         d_l1 = []
#         for _ in range(n):
#             d_l1.append(common.BBlock(conv, 160, 160, 3, act=act))
#
#         # downsample的第二层，维度变化640->256（默认的feature map == 64）
#         d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3, act=act)]
#         for _ in range(n):
#             d_l2.append(
#                 common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
#
#         # downsample的第三层，并与upsample进行连接，也是upsample的第三层
#         # 维度变化1024->256，256->1024
#         pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]
#         for _ in range(n * 2):
#             pro_l3.append(
#                 common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
#         pro_l3.append(
#             common.BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))
#
#         # upsample的第二层，维度变化256->640
#         i_l2 = []
#         for _ in range(n):
#             i_l2.append(
#                 common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
#         i_l2.append(common.BBlock(conv, n_feats * 4, 640, 3, act=act))
#
#         # upsample的第一层，维度变化160->4
#         i_l1 = []
#         for _ in range(n):
#             i_l1.append((common.BBlock(conv, 160, 160, 3, act=act)))
#         m_tail = [conv(160, 4, 3)]
#
#         # downsample的第一层
#         self.head = nn.Sequential(*m_head)
#         self.d_l1 = nn.Sequential(*d_l1)
#         # downsample的第二层
#         self.d_l2 = nn.Sequential(*d_l2)
#         # 第三层连接层
#         self.pro_l3 = nn.Sequential(*pro_l3)
#         # upsample的第二层
#         self.i_l2 = nn.Sequential(*i_l2)
#         # upsample的第一层
#         self.i_l1 = nn.Sequential(*i_l1)
#         self.tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         # downsample的第一层
#         x1 = self.d_l1(self.head(self.DWT(x)))
#         # downsample的第二层
#         x2 = self.d_l2(self.DWT(x1))
#         # upsample的第三层，并且使用了short cut的结构，将对应的downsample的第二层
#         # 加到upsample对应的层上。
#         x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
#         # upsample的第二层
#         x_ = self.IWT(self.i_l2(x_)) + x1
#         # upsample的第一层
#         x = self.IWT(self.tail(self.i_l1(x_))) + x
#         # x = self.add_mean(x)
#         return x
#
#     def set_scale(self, scale_idx):
#         self.scale_idx = scale_idx
#
#     def set_scale(self, scale_idx):
#         self.scale_idx = scale_idx