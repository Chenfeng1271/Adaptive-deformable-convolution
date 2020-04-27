import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.DenseDSPP import build_densedspp
from modeling.decoder_gau_v2 import build_decoder_gau
from modeling.dspp import build_dspp
from modeling.fpa import build_fpa
from modeling.DenseDSPP_v3 import build_densedspp_v3
from modeling.DenseASPP import build_DenseASPP
from modeling.stack_conv_resudial import build_stack_resudial_conv
from modeling.stack_conv import build_stack_conv

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)#输出三个  x, feature_map,low_level_feat
        #self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        #self.dspp=build_dspp(backbone,output_stride,BatchNorm,modulation=False,adaptive_d= False)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        #self.baseline=nn.Sequential(nn.Conv2d(in_channels=2048,out_channels=256,kernel_size=1,stride=1),BatchNorm(256))
        #self.denseaspp = build_DenseASPP(BatchNorm)
        self.stack_resudial = build_stack_resudial_conv(backbone,output_stride,BatchNorm=BatchNorm,modulation=False,adaptive_d=False,deform=True)
        #self.stack = build_stack_conv(backbone,output_stride,modulation=True,adaptive_d=False,BatchNorm=BatchNorm,deform=True)
        #self.densedspp = build_densedspp()
        #self.densedspp_v3 =build_densedspp_v3(modulation=False,adaptive_d = False)
        #self.decoder_gau =build_decoder_gau(BatchNorm)
        #self.fpa = build_fpa(2048)
        #self.conv3x3_dspp_decoder = nn.Conv2d(2048,256,3)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x,feature_map,low_level_feat = self.backbone(input)
        
        #x = self.aspp(x)
        #x = self.stack(x)
        x = self.stack_resudial(x)
        #x=self.dspp(x)
        #x = self.denseaspp(x)
        #x =self.densedspp_v3(x)
        #x = self.densedspp(x)
        #x=self.baseline(x)
        #x = self.conv3x3_dspp_decoder(x)
        #x =self.fpa(x)
        #x = self.decoder_gau(x, feature_map[::-1])
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.stack_resudial, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


