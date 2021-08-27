import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# __all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
#            'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def make_model(args, parent=False):
    return Model(args)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out
        out = self.relu(out)

        return out



class ResNet_iqa(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 96
        super(ResNet_iqa, self).__init__()

        self.layer1 = self._make_layer(block,  96, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3])
        #self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):


        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.leaky = 0.1

        self.group_layers = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=2),
            nn.ReLU(inplace=True)
        )

        self.shared_layers1 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(self.leaky, inplace=True)
        )
        self.shared_layers2 = ResNet_iqa(BasicBlock, [3, 4, 6, 3])
        self.shared_layers3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky, inplace=True))

        self.score_layer1 = nn.Sequential(
            nn.Conv2d(384, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky, inplace=True)
        )

        self.score_layer2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )

    def forward(self, x, y):  # pylint: disable=W0221
        """
        :param x: Impaired viewports. shape: (batch_size, channels, height, width)
        :param y: Viewport error map with the same shape of x.
        :return:
        """
        x = torch.cat((x, y), dim=1) # 2, 256, 256
        #x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        batch_size = x.shape[0]
        x = self.group_layers(x) # 64, 256, 256
        # print(x.size())
        # print(self.shared_layers1(x).size())
        # print(self.shared_layers2(self.shared_layers1(x)).size())
        x = self.shared_layers3(self.shared_layers2(self.shared_layers1(x)))


        z = nn.functional.interpolate(y, x.shape[-2:], mode='bilinear')
        z = z.repeat(1, 128, 1, 1)
        x = torch.cat((x, z), dim = 1)
        x = self.score_layer1(x)

        x1 = F.max_pool2d(x, kernel_size=2, stride=2)
        x2 = F.avg_pool2d(x, kernel_size=2, stride=2)
        x12 = torch.cat((x1, x2), dim = 1)
        x12 = self.score_layer2(x12)

        x11 = F.max_pool2d(x12, kernel_size=2, stride=2)
        x21 = F.avg_pool2d(x12, kernel_size=2, stride=2)
        x22 = torch.cat((x11, x21), dim = 1)

        x22 = self.pool(x22)

        x22 = x22.view(batch_size, -1)
        x22 = self.fc(x22)
        return x22

#
# def resnet34_cbam(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
#         now_state_dict        = model.state_dict()
#         now_state_dict.update(pretrained_state_dict)
#         model.load_state_dict(now_state_dict)
#     return model





