# Code for Cross-Neuron Communication Network
# Author: Jianwei Yang (jw2yang@gatech.edu)
import math
import torch
import torch.nn as nn
import torchvision.models as models

from .networks import *
from .layers import *

class CrossNeuronNetAnalysis(nn.Module):
    def __init__(self, opts):
        super(CrossNeuronNetAnalysis, self).__init__()
        if opts.dataset == "imagenet": # we directly use pytorch version arch
            import pdb; pdb.set_trace()
            if opts.arch in models.__dict__:
                self.net = models.__dict__[opts.arch](pretrained=False)
            elif opts.arch in imagenet_models:
                self.net = imagenet_models[opts.arch](num_classes=1000)
            else:
                raise ValueError("Unknow network architecture for imagenet, please refer to: https://pytorch.org/docs/0.4.0/torchvision/models.html?")
            if opts.add_cross_neuron:
                add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 24, 24, 16], [32, 24, 24, 16], reduction = [4, 4, 4, 2])
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
                self.net = cifar_models[opts.arch](num_classes=100, insert_layers=opts.layers, depth=opts.depth,
                        has_selayer=("se" in opts.arch), has_gtlayer=opts.add_cross_neuron, communication=opts.communication)
            else:
                raise ValueError("Unknow network architecture for cifar100")
            # if opts.add_cross_neuron:
                # add_cross_neuron(self.net, opts.data.input_size, opts.data.input_size, [32, 24, 24, 16], [32, 24, 24, 16])
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
        out = self.net(x)
        return out
