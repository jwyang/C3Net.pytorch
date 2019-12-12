'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math

from lib.layers import *

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class _CrossNeuronBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width,
                    nblocks_channel=8,
                    spatial_height=32, spatial_width=32,
                    reduction=8, size_is_consistant=True,
                    communication=True,
                    enc_dec=True):
        # nblock_channel: number of block along channel axis
        # spatial_size: spatial_size
        super(_CrossNeuronBlock, self).__init__()

        self.communication = communication
        self.enc_dec = enc_dec

        # set channel splits
        if in_channels <= 512:
            self.nblocks_channel = 1
        else:
            self.nblocks_channel = in_channels // 512
        block_size = in_channels // self.nblocks_channel
        block = torch.Tensor(block_size, block_size).fill_(1)
        self.mask = torch.Tensor(in_channels, in_channels).fill_(0)
        for i in range(self.nblocks_channel):
            self.mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size].copy_(block)

        # set spatial splits
        if in_height * in_width < 16 * 16 and size_is_consistant:
            self.spatial_area = in_height * in_width
            self.spatial_height = in_height
            self.spatial_width = in_width
        else:
            self.spatial_area = spatial_height * spatial_width
            self.spatial_height = spatial_height
            self.spatial_width = spatial_width

        self.fc_in = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
        )

        self.fc_out = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
        )

        self.bn = nn.BatchNorm1d(self.spatial_area)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        :param x: (bt, c, h, w)
        :return:
        '''
        bt, c, h, w = x.shape
        residual = x
        x_stretch = x.view(bt, c, h * w)
        spblock_h = int(np.ceil(h / self.spatial_height))
        spblock_w = int(np.ceil(w / self.spatial_width))
        stride_h = int((h - self.spatial_height) / (spblock_h - 1)) if spblock_h > 1 else 0
        stride_w = int((w - self.spatial_width) / (spblock_w - 1)) if spblock_w > 1 else 0

        if self.spatial_height == h and self.spatial_width == w:
            x_stacked = x_stretch # (b) x c x (h * w)
            x_stacked = x_stacked.view(bt * self.nblocks_channel, c // self.nblocks_channel, -1)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c

            if self.enc_dec:
                x_v = self.fc_in(x_v) # (b) x (h * w) x c

            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel) # (b * h * w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * h * w) x c x c
            # score = torch.bmm(x_v.transpose(1, 2).contiguous(), x_v)
            # x_v = F.dropout(x_v, 0.1, self.training)
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c

            if self.communication:
                if self.enc_dec:
                    out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b) x (h * w) x c
                else:
                    out = self.bn(torch.bmm(x_v, attn)) # (b) x (h * w) x c
            else:
                out = self.bn(self.fc_out(x_v)) # (b) x (h * w) x c

            out = out.permute(0, 2, 1).contiguous().view(bt, c, h, w)
            return (residual + out)
        else:
            x = F.interpolate(x, (self.spatial_height, self.spatial_width))
            x_stretch = x.view(bt, c, self.spatial_height * self.spatial_width)
            x_stretch = x.view(bt * self.nblocks_channel, c // self.nblocks_channel, self.spatial_height * self.spatial_width)

            x_stacked = x_stretch # (b) x c x (h * w)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c

            if self.enc_dec:
                x_v = self.fc_in(x_v) # (b) x (h * w) x c

            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel) # (b * h * w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * h * w) x c x c
            # score = torch.bmm(x_v.transpose(1, 2).contiguous(), x_v)
            # x_v = F.dropout(x_v, 0.1, self.training)
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c
            if self.communication:
                if self.enc_dec:
                    out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b) x (h * w) x c
                else:
                    out = self.bn(torch.bmm(x_v, attn)) # (b) x (h * w) x c
            else:
                out = self.bn(self.fc_out(x_v)) # (b) x (h * w) x c
            out = out.permute(0, 2, 1).contiguous().view(bt, c, self.spatial_height, self.spatial_width)
            out = F.interpolate(out, (h, w))
            return (residual + out)

class BasicBlockPlain(nn.Module):
    expansion=1

    def __init__(self, insize, inplanes, planes, stride=1, downsample=None, use_se=False, reduction=8):
        super(BasicBlockPlain, self).__init__()
        self.use_se = use_se
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.use_se:
            self.se = SELayer(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # out = self.gt_spatial(out)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, insize, inplanes, planes, stride=1, downsample=None, use_se=False, use_gt=False, reduction=8):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        self.use_gt = use_gt
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # if not self.use_gt:
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)
        if self.use_se:
            self.se = SELayer(planes, reduction=reduction)
        if self.use_gt:
            self.gt = _CrossNeuronBlock(planes, insize, insize, spatial_height=16, spatial_width=16, reduction=8)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # out = self.gt_spatial(out)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gt:
            out = self.gt(out)
            
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, insize, inplanes, planes, stride=1, downsample=None, use_se=False, use_gt=False, reduction=8):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        self.use_gt = use_gt
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        if self.use_se:
            self.se = SELayer(planes, reduction=reduction)
        if self.use_gt:
            self.gt = CrossNeuronlBlock2D(planes, insize, insize, insize, insize, reduction=8)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)
        if self.use_gt:
            out = self.gt(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# class CommunicationBlock(nn.Module):


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, insert_layers=["1", "2", "3"], depth=1,
            has_gtlayer=False, has_selayer=False, communication=True, enc_dec=True):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.insize = 32
        self.layers = insert_layers
        self.depth = depth
        self.has_gtlayer = has_gtlayer
        self.has_selayer = has_selayer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], has_selayer=has_selayer,
            has_gtlayer=(has_gtlayer if ("1" in self.layers) else False))
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, has_selayer=has_selayer,
            has_gtlayer=(has_gtlayer if ("2" in self.layers) else False))
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, has_selayer=has_selayer,
            has_gtlayer=(has_gtlayer if ("3" in self.layers) else False))

        # if self.has_gtlayer:
        #     if "1" in self.layers:
        #         layers = []
        #         for i in range(self.depth):
        #             layers.append(_CrossNeuronBlock(16, 32, 32, spatial_height=16, spatial_width=16,
        #                 communication=communication, enc_dec=enc_dec))
        #         self.nclayer1 = nn.Sequential(*layers)
        #
        #     if "2" in self.layers:
        #         layers = []
        #         for i in range(self.depth):
        #             layers.append(_CrossNeuronBlock(16, 32, 32, spatial_height=16, spatial_width=16,
        #                 communication=communication, enc_dec=enc_dec))
        #         self.nclayer2 = nn.Sequential(*layers)
        #
        #     if "3" in self.layers:
        #         layers = []
        #         for i in range(self.depth):
        #             layers.append(_CrossNeuronBlock(32, 16, 16, spatial_height=16, spatial_width=16,
        #                 communication=communication, enc_dec=enc_dec))
        #         self.nclayer3 = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, has_selayer=False, has_gtlayer=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.insize, self.inplanes, planes, stride, downsample, use_se=has_selayer, use_gt=has_gtlayer))
        self.inplanes = planes * block.expansion
        self.insize = int(self.insize / stride)
        for _ in range(1, blocks - 1):
            layers.append(block(self.insize, self.inplanes, planes, use_se=has_selayer, use_gt=False))
        layers.append(block(self.insize, self.inplanes, planes, use_se=has_selayer, use_gt=False))
        return nn.Sequential(*layers)

    def _correlation(self, x):
        b, c, h, w = x.shape
        x_v = x.clone().detach().view(b, c, -1) # b x c x (hw)
        x_m = x_v.mean(1).unsqueeze(1) # b x 1 x (hw)
        x_c = x_v - x_m # b x c x (hw)
        # x_c = x_v
        num = torch.bmm(x_c, x_c.transpose(1, 2)) # b x c x c
        x_mode = torch.sqrt(torch.sum(x_c ** 2, 2).unsqueeze(2)) # b x c x 1
        dec = torch.bmm(x_mode, x_mode.transpose(1, 2).contiguous()) # b x c x c
        out = num / dec
        out = torch.abs(out)
        return out.mean()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x0 = x.clone().detach()
        corr1 = self._correlation(x)

        x = self.layer1(x)
        x1 = x.clone().detach()
        corr2 = self._correlation(x)

        x = self.layer2(x)
        x2 = x.clone().detach()
        corr3 = self._correlation(x)

        x = self.layer3(x)
        x3 = x.clone().detach()
        corr4 = self._correlation(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, (x0, x1, x2, x3), (corr1.item(), corr2.item(), corr3.item(), corr4.item()), (corr1.item(), corr2.item(), corr3.item(), corr4.item())


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, has_gtlayer=False, has_selayer=False):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.insize = 32
        self.has_gtlayer = has_gtlayer
        self.has_selayer = has_selayer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet20plain_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlockPlain, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet62a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [10, 10, 10], **kwargs)
    return model

def resnet68a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [11, 11, 11], **kwargs)
    return model

def resnet74a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [12, 12, 12], **kwargs)
    return model

def resnet80a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [13, 13, 13], **kwargs)
    return model

def resnet86a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [14, 14, 14], **kwargs)
    return model

def resnet92a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [15, 15, 15], **kwargs)
    return model

def resnet98a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [16, 16, 16], **kwargs)
    return model

def resnet104a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [17, 17, 17], **kwargs)
    return model

def resnet110a_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet110plain_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlockPlain, [18, 18, 18], **kwargs)
    return model

def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
