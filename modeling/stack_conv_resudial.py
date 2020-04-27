import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from modeling.deformable_conv.deform_conv_v3 import *


class _stack_Module(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm,modulation,adaptive_d,deform):
        super(_stack_Module, self).__init__()
        self.deform_tag=deform

        if self.deform_tag==True:
            self.deform_conv = DeformConv2d(inplanes,planes,kernel_size=kernel_size,stride=1,padding=padding,modulation=modulation,bias=False,adaptive_d=adaptive_d,dilation=dilation)
        else:
            self.main_conv= nn.Conv2d(inplanes,planes,kernel_size=kernel_size,stride=1,padding=padding,bias=False,dilation=dilation)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self.side_branch = nn.Conv2d(inplanes,planes,1)

        self._init_weight()

    def forward(self, x):
        x_side = self.side_branch(x)
        if self.deform_tag==True:

            x = self.deform_conv(x)
        else:
            x = self.main_conv(x)
        x = self.bn(x)

        return self.relu(x_side+x)

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

class Stack_Conv(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm,modulation=True,adaptive_d=True,deform=True):
        super(Stack_Conv, self).__init__()
        self.modulation=modulation
        self.deform_all = deform
        self.adaptive_all =adaptive_d
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.dspp1 = _stack_Module(inplanes, 1024, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_all,deform=self.deform_all)
        self.dspp2 = _stack_Module(1024, 1024, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_all,deform=self.deform_all)
        self.dspp3 = _stack_Module(1024, 512, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_all,deform=self.deform_all)
        self.dspp4 = _stack_Module(512, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm,modulation=self.modulation,adaptive_d=self.adaptive_all,deform=self.deform_all)


        self.conv1 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x = self.dspp1(x)
        x = self.dspp2(x)
        x = self.dspp3(x)
        x = self.dspp4(x)

        #x = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=True)
        #x = torch.cat((x1, x2, x3, x4, x5), dim=1)

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


def build_stack_resudial_conv(backbone, output_stride, BatchNorm,modulation=True,adaptive_d=True,deform=True):
    return Stack_Conv(backbone, output_stride, BatchNorm,modulation=modulation,adaptive_d=adaptive_d,deform=deform)  