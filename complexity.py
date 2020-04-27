from torchvision.models import resnet50
from thop import profile
import torch

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

from modeling.DenseDSPP import build_dspp
from modeling.decoder_gau import build_decoder_gau

dspp =build_dspp().cuda()
flops,params=profile(dspp,input_size=(1,2048,16,16))
#decoder_gau=build_decoder_gau(nn.BatchNorm2d).cuda()
#flops,params=profile(decoder_gau,input_size=(1,2048,16,16))
print(flops,params)