# pytorch-classification-cifar
practice on cifar or custom datasets using pytorch

## Requirements
This is my experiment eviroument：<br>

numpy==1.24.2<br>
opencv_python==4.6.0.66<br>
Pillow==9.4.0<br>
timm==0.6.11<br>
torch==1.12.0+cu113<br>
torchvision==0.13.0+cu113<br>
tqdm==4.59.0

## Usage
### 1. enter directory
`$ cd pytorch-classification-cifar`

### 2.dataset
I use cifar10 and cifar100 dataset from torchvision, and you can create your own dataset too.<br>
Default use cifar10 dataset. If you want to use cifar100 dataset, you just need to change the "cifar10" in ./utils/dataloaders.py to "cifar100".

### 3. train the model
You need to adjust the parameters you need. Default use gpu to train resnet50 with cifar10.<br>
`$ python train.py`<br><br>
You can warmup training by set -warm to 1 or 2, to prevent network diverge during early training phase.<br><br>
The supported net args are:<br>
squeezenet<br>
mobilenet<br>
mobilenetv2<br>
shufflenet<br>
shufflenetv2<br>
vgg<br>
densenet<br>
googlenet<br>
inceptionv3<br>
inceptionv4<br>
inceptionresnetv2<br>
xception<br>
resnet<br>
preactresnet<br>
resnext<br>
attention<br>
seresnet<br>
nasnet<br>
wideresnet<br>
stochasticdepth<br>

The training will save two weight files with suffixes "best" and "last".

### 4. test the model
You need to adjust the parameters you need with using test.py.<br>
`$ python test.py -net resnet50 -weights path_to_resnet50_weights_file`
