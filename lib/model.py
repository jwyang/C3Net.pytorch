# Code for Cross-Neuron Communication Network
# Author: Jianwei Yang (jw2yang@gatech.edu)
import math
import torch
import torch.nn as nn
import torchvision.models as models

from .networks import *
from .layers import *
from .layers.cross_neuron_distributed import _CrossNeuronBlock

class AlexNet(nn.Module):
    def __init__(self, num_classes=100, has_gtlayer=False):
        super(AlexNet, self).__init__()

        if has_gtlayer:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                _CrossNeuronBlock(64, 8, 8, spatial_height=8, spatial_width=8, reduction=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.has_gtlayer = has_gtlayer

        # if has_gtlayer:
            # self.gtlayer = _CrossNeuronBlock(256, 16, 16, 16, 16)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CrossNeuronNet(nn.Module):
    def __init__(self, opts):
        super(CrossNeuronNet, self).__init__()
        if opts.dataset == "imagenet": # we directly use pytorch version arch
            if opts.arch in models.__dict__:
                self.net = models.__dict__[opts.arch](pretrained=False)
                if opts.arch == "vgg16":
                    for i in range(len(self.net.features)):
                        if self.net.features[i].__class__.__name__ == "Conv2d":
                            channels = self.net.features[i].out_channels
                            self.net.features[i] = nn.Sequential(
                                self.net.features[i],
                                nn.BatchNorm2d(channels)
                            )
            elif opts.arch in imagenet_models:
                self.net = imagenet_models[opts.arch](num_classes=1000)
            else:
                raise ValueError("Unknow network architecture for imagenet, please refer to: https://pytorch.org/docs/0.4.0/torchvision/models.html?")
            # if opts.add_cross_neuron:
                # add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 24, 24, 16], [32, 24, 24, 16], reduction = [4, 4, 4, 2])
                # add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 24, 24, 16], [32, 24, 24, 16])

        elif opts.dataset == "cub2011":
            if opts.arch in models.__dict__:
                self.net = models.__dict__[opts.arch](pretrained=True)
                self.net.fc = nn.Linear(self.net.fc.in_features, 200, bias=True)
                # NOTE: replace the last fc layer
            else:
                raise ValueError("Unknow network architecture for cub2011, please refer to: https://pytorch.org/docs/0.4.0/torchvision/models.html?")
            if opts.add_cross_neuron:
                add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 32, 32, 16], [32, 32, 32, 16], 8)
        elif opts.dataset == "cifar10": # we use the arch following He's paper (deep residual learning)
            if opts.arch in cifar_models:
                self.net = cifar_models[opts.arch](num_classes=10, has_selayer=("se" in opts.arch))
            else:
                raise ValueError("Unknow network architecture for imagenet")
            if opts.add_cross_neuron:
                add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 24, 24, 16], [32, 24, 24, 16])
        elif opts.dataset == "cifar100": # we use the arch following He's paper (deep residual learning)
            if opts.arch in cifar_models:
                self.net = cifar_models[opts.arch](num_classes=100, has_selayer=("se" in opts.arch), has_gtlayer=False)
            elif opts.arch == "alexnet":
                self.net = AlexNet(has_gtlayer=opts.add_cross_neuron)
            else:
                raise ValueError("Unknow network architecture for cifar100")
            self.net.name = "cnn"
            if opts.add_cross_neuron and not opts.arch == "alexnet":
                add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [16, 16, 8, 8], [16, 16, 8, 8])
        else:
            raise ValueError("Unknow dataset, we only support cifar and imagenet for now")

        print(self.net)

    def get_optim_policies(self):
        resnet_param = []
        ccn_param = []
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.BatchNorm1d):
                ccn_param = ccn_param + list(m.parameters())
            else:
                resnet_param = resnet_param + list(m.parameters())

        return [{"params": resnet_param, 'lr_mult': 1}, {"params": ccn_param, 'lr_mult': 10}]

    def forward(self, x):
        # x = self.net.conv1(x)
        # x = self.net.bn1(x)
        # x = self.net.relu(x)
        # # x = self.net.maxpool(x)
        # x0 = x.clone().detach()
        #
        # x = self.net.layer1(x)
        # x1 = x.clone().detach()
        #
        # x = self.net.layer2(x)
        # x2 = x.clone().detach()
        #
        # x = self.net.layer3(x)
        # x3 = x.clone().detach()

        # x = self.net.layer4(x)
        # x4 = x.clone().detach()

        # out = self.net.avgpool(x).squeeze(3).squeeze(2)
        # out = self.net.fc(out)
        out = self.net(x)
        return out
