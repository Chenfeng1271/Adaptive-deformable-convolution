
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.deformable_conv.deform_conv_v3 import *
from torch.nn import BatchNorm2d as bn


class _DenseAsppBlock(nn.Module):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True,modulation=True,adaptive_d=True):
        super(_DenseAsppBlock, self).__init__()
        self.modulation = modulation
        self.adaptive_d = adaptive_d
        self.bn_start = bn_start
        self.bn1 = bn(input_num, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv_1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.bn2 = bn(num1, momentum=0.0003)
        self.relu2 = nn.ReLU(inplace = True)
        
        self.deform_conv = DeformConv2d(num1,num2,3,padding=1,dilation=dilation_rate,modulation=self.modulation,adaptive_d=self.adaptive_d)
        
        #self.offset = ConvOffset2D(num1)
        #self.conv_2 = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,padding=1)
        #self.conv_3 =nn.Conv2d(in_channels=num2, out_channels=num2, kernel_size=3,dilation=dilation_rate,
        #                       padding=dilation_rate)

    def forward(self,input):
        if self.bn_start == True:
            input = self.bn1(input)
        feature = self.relu1(input)
        feature = self.conv_1(feature)
        feature = self.bn2(feature)
        #feature1 =self.offset(feature)
        #feature1 = self.conv_2(feature1)
        
        feature1 = self.deform_conv(feature)

        #feature1 = self.conv_3(feature1)
        #feature2 = self.conv_3(feature)
        #feature3 = feature1 + feature2
        return feature1



class DenseASPP(nn.Module):
    def __init__(self,num_features,d_feature0,d_feature1,dropout0,modulation=True,adaptive_d=True):
        super(DenseASPP,self).__init__()
        self.num_features = num_features
        self.d_feature0 = d_feature0
        self.d_feature1 = d_feature1
        self.init_feature = 2048 - 5*d_feature1
        self.dropout0 = dropout0
        self.adaptive_all = adaptive_d
        self.modulation_all = modulation 

        self.init_conv_aspp = nn.Conv2d(self.num_features,self.init_feature,kernel_size=(3,3),padding=1)
        self.num_features = self.init_feature

        self.ASPP_3 = _DenseAsppBlock(input_num=self.num_features, num1=self.d_feature0, num2=self.d_feature1,
                                      dilation_rate=3, drop_out=self.dropout0, bn_start=False,modulation= self.modulation_all,adaptive_d=self.adaptive_all)

        self.ASPP_6 = _DenseAsppBlock(input_num=self.num_features + self.d_feature1 * 1, num1=self.d_feature0, num2=self.d_feature1,
                                      dilation_rate=6, drop_out=self.dropout0, bn_start=True,modulation= self.modulation_all,adaptive_d=self.adaptive_all)

        self.ASPP_12 = _DenseAsppBlock(input_num=self.num_features + self.d_feature1 * 2, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=12, drop_out=self.dropout0, bn_start=True,modulation= self.modulation_all,adaptive_d=self.adaptive_all)

        self.ASPP_18 = _DenseAsppBlock(input_num=self.num_features + self.d_feature1 * 3, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=18, drop_out=self.dropout0, bn_start=True,modulation= self.modulation_all,adaptive_d=self.adaptive_all)

        self.ASPP_24 = _DenseAsppBlock(input_num=self.num_features + self.d_feature1 * 4, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=24, drop_out=self.dropout0, bn_start=True,modulation= self.modulation_all,adaptive_d=self.adaptive_all)

    def forward(self,feature):

            feature = self.init_conv_aspp(feature)
            aspp3 = self.ASPP_3(feature)
            feature = torch.cat((aspp3, feature), dim=1)

            aspp6 = self.ASPP_6(feature)
            feature = torch.cat((aspp6, feature), dim=1)

            aspp12 = self.ASPP_12(feature)
            feature = torch.cat((aspp12, feature), dim=1)

            aspp18 = self.ASPP_18(feature)
            feature = torch.cat((aspp18, feature), dim=1)

            aspp24 = self.ASPP_24(feature)
            feature = torch.cat((aspp24, feature), dim=1)
            return feature

def build_densedspp_v3(modulation=True,adaptive_d=True):
    return DenseASPP(2048,512,128,.1,modulation = modulation,adaptive_d=adaptive_d)
