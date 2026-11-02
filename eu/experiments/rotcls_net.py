# rotcls_net.py -- ResNet-18 with the CIFAR stem (3x3 convolution, stride 1, no max-pool) for the rotation classifiers
import torch.nn as nn
import torchvision


def build(ch, n_out):
    net = torchvision.models.resnet18(num_classes=n_out)
    net.conv1 = nn.Conv2d(ch, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    return net
