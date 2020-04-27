import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.deformable_conv.deform_conv_v2 import *

class Deform_GAU(nn.Module):
    def __init__(self,channels_high,channels_low,upsample=True):
        super(Deform_GAU,self).__init__()

        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low,channels_low,kernel_size=3,padding=1,bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)
        self.deform3x3 = DeformConv2d(channels_low,channels_low,kernel_size = 3,padding=1,modulation=True)
        self.bn_high = nn.BatchNorm2d(channels_low)
        self.deform_sigmoid = nn.Sigmoid()

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,fms_high,fms_low,fm_mask=None):
        fms_low = self.conv3x3(fms_low)
        fms_low = self.bn_low(fms_low)

        if self.upsample:
            fms_high = self.conv_upsample(fms_high)
            fms_high = self.bn_upsample(fms_high)
        else:
            fms_high = self.conv_reduction(fms_high)
            fms_high = self.bn_reduction(fms_high)

        fms_high_copy = fms_high
        fms_high,m_mask = self.deform3x3(fms_high_copy)
        m_mask = self.deform_sigmoid(m_mask)
        fms_high = self.bn_high(fms_high)
        fms_att = fms_low*m_mask +fms_high

        out = self.relu(fms_att)

        return out

class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        #fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out


class EX_GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True,BatchNorm=nn.BatchNorm2d):
        super(EX_GAU, self).__init__()
        # Global Attention Upsample
        
        self.upsample = upsample
        
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = BatchNorm(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = BatchNorm(channels_low)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = BatchNorm(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        #fms_low = fms_low+fms_seb
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        #fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            out = self.relu(
                self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out

class SEB(nn.Module):
    def __init__(self, channels_low,BatchNorm=nn.BatchNorm2d):
        super(SEB,self).__init__()
        self.conv1x1 = nn.Conv2d(channels_low, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high =BatchNorm(channels_low)
        self.relu= nn.ReLU(inplace=True)
        self.conv3x3_gp= nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)

        self.conv3x3_seb =nn.Conv2d(channels_low,int(channels_low/2),kernel_size=3,padding=1,stride=1,bias=False)
        self.bn_seb =BatchNorm(int(channels_low/2))

    def forward(self, fms_high,fms_low):
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask =fms_low*fms_high_gp
        fms_low_mask=self.conv3x3_gp(fms_low_mask)
        fms_low_mask=fms_low_mask+fms_high

        fms_seb= self.conv3x3_seb(fms_low_mask)
        fms_seb = F.interpolate(fms_seb, size=tuple(np.array(fms_low.size()[2:])*2), mode='bilinear', align_corners=True)
        fms_seb=self.bn_seb(fms_seb)
        fms_seb=self.relu(fms_seb)

        return fms_low_mask,fms_seb
        


class Decoder_GAU(nn.Module):
    def __init__(self,BatchNorm):
        super(Decoder_GAU,self).__init__()
        channels_blocks = [2048, 1024, 512, 256]
        self.gau_block1 = EX_GAU(channels_blocks[0], channels_blocks[1], upsample=False)
        self.gau_block2 = EX_GAU(channels_blocks[1], channels_blocks[2])
        self.gau_block3 = EX_GAU(channels_blocks[2], channels_blocks[3],upsample=True)
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3]

        self.seb_block1=SEB(channels_blocks[1])
        self.seb_block2=SEB(channels_blocks[2])
        self.seb_block3=SEB(channels_blocks[3])
        self.seb=[self.seb_block1,self.seb_block2,self.seb_block3]
        self.conv3x3_aspp_upsample =nn.Conv2d(channels_blocks[0],channels_blocks[1],kernel_size=3,padding=1,stride=1,bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.mask_conv = nn.Conv2d(256,21, kernel_size=3, stride=1, padding=1)

    def forward(self,x,fms=[]):
        """
        :param fms: Feature maps of forward propagation in the network with reverse sequential. shape:[b, c, h, w]
        :return: fm_high. [b, 256, h, w]
        """

        ####################测试FLOPs和param#################
        #a=torch.randn(1,2048,16,16)
        #b=torch.randn(1,1024,16,16)
        #c=torch.randn(1,512,32,32)
        #d = torch.randn(1,256,64,64)
        #fms=[a,b,c,d]
        ################################################
        for i, fm_low in enumerate(fms):
            if i == 0:
                
                fm_seb =self.conv3x3_aspp_upsample(fm_low)
                fm_high = x
                #fm_high = fm_low
            else:
                fm_low_mask ,fm_seb =self.seb[int(i-1)](fm_low,fm_seb)
                fm_high= self.gau[int(i-1)](fm_high, fm_low_mask)
        out = self.mask_conv(fm_high)

        return out
def build_decoder_gau(BatchNorm):
    return Decoder_GAU(BatchNorm)