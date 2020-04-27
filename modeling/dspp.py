import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.deformable_conv.deform_conv_v3 import *

class _DSPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm,modulation,adaptive_d):
        super(_DSPPModule, self).__init__()

        self.deform_conv = DeformConv2d(inplanes,planes,kernel_size=kernel_size,stride=1,padding=padding,modulation=modulation,adaptive_d=adaptive_d,bias=False,dilation=dilation)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x= self.deform_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DSPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm,modulation=True,adaptive_d=True):
        super(DSPP, self).__init__()
        self.modulation=modulation
        self.adaptive_d = adaptive_d
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]#1 6 12 18
        elif output_stride == 8:
            dilations = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        self.dspp1 = _DSPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_d)
        self.dspp2 = _DSPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_d)
        self.dspp3 = _DSPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_d)
        self.dspp4 = _DSPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_d)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.dspp1(x)
        x2 = self.dspp2(x)
        x3 = self.dspp3(x)
        x4 = self.dspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_dspp(backbone, output_stride, BatchNorm,modulation=True,adaptive_d=True):
    return DSPP(backbone, output_stride, BatchNorm,modulation=modulation,adaptive_d=adaptive_d)
