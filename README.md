# pytorch-adaptive-deformable-convolution



### Introduction

This is a PyTorch(0.4.1) implementation of Adaptive deformable convnet. For the result please refer to our paper under review.


### Installation

The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

0. Clone the repo:

1. Install dependencies:

   For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

   For custom dependencies:

   ```Shell
   pip install matplotlib pillow tensorboardX tqdm
   ```

### Training

Fellow steps below to train your model:

0. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py).

1. Input arguments: (see full input arguments via python train.py --help):

   ```Shell
   usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
               [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
               [--use-sbd] [--workers N] [--base-size BASE_SIZE]
               [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
               [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
               [--start_epoch N] [--batch-size N] [--test-batch-size N]
               [--use-balanced-weights] [--lr LR]
               [--lr-scheduler {poly,step,cos}] [--momentum M]
               [--weight-decay M] [--nesterov] [--no-cuda]
               [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
               [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
               [--no-val]
   
   ```

   

   

### Acknowledgement

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn]

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[mmdetection](https://github.com/open-mmlab/mmdetection)

[Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)