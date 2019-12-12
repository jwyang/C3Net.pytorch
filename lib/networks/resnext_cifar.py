"""
resneXt for cifar with pytorch

Reference:
[1] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017
"""

import torch
import torch.nn as nn
import math

from lib.layers import *

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

            out = F.dropout(out.permute(0, 2, 1).contiguous().view(bt, c, h, w), 0.0, self.training)
            return F.relu(residual + out)
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
            out = F.dropout(F.interpolate(out, (h, w)), 0.0, self.training)
            return F.relu(residual + out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, baseWidth, stride=1, downsample=None, use_se=False):
        super(Bottleneck, self).__init__()
        D = int(planes * (baseWidth / 64.))
        C = cardinality
        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.se = nn.SELayer(planes*4, reduction=8)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size() != out.size():
            print(out.size(), residual.size())
        out += residual
        out = self.relu(out)

        return out


class ResNeXt_Cifar(nn.Module):

    def __init__(self, block, layers, cardinality, baseWidth, num_classes=10, insert_layers=["1", "2", "3"], depth=1,
            has_gtlayer=False, has_selayer=False, communication=True, enc_dec=True):
        super(ResNeXt_Cifar, self).__init__()
        self.inplanes = 64

        self.insize = 32
        self.layers = insert_layers
        self.depth = depth
        self.has_gtlayer = has_gtlayer
        self.has_selayer = has_selayer

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        if self.has_gtlayer:
            if "1" in self.layers:
                layers = []
                for i in range(self.depth):
                    layers.append(_CrossNeuronBlock(64, 32, 32, spatial_height=16, spatial_width=16,
                        communication=communication, enc_dec=enc_dec))
                self.nclayer1 = nn.Sequential(*layers)

            if "2" in self.layers:
                layers = []
                for i in range(self.depth):
                    layers.append(_CrossNeuronBlock(64, 32, 32, spatial_height=16, spatial_width=16,
                        communication=communication, enc_dec=enc_dec))
                self.nclayer2 = nn.Sequential(*layers)

            if "3" in self.layers:
                layers = []
                for i in range(self.depth):
                    layers.append(_CrossNeuronBlock(128, 16, 16, spatial_height=8, spatial_width=8,
                        communication=communication, enc_dec=enc_dec))
                self.nclayer3 = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth, stride, downsample, use_se=self.has_selayer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.baseWidth, use_se=self.has_selayer))

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

        corr1 = self._correlation(x)
        x0 = x.clone().detach()

        if self.has_gtlayer and "1" in self.layers:
            x = self.nclayer1(x)
            corr1_ = self._correlation(x)
        else:
            corr1_ = corr1.clone()

        x = self.layer1(x)
        x1 = x.clone().detach()

        corr2 = self._correlation(x)
        if self.has_gtlayer and "2" in self.layers:
            x = self.nclayer2(x)
            corr2_ = self._correlation(x)
        else:
            corr2_ = corr2.clone()

        x = self.layer2(x)
        x2 = x.clone().detach()

        corr3 = self._correlation(x)
        if self.has_gtlayer and "3" in self.layers:
            x = self.nclayer3(x)
            corr3_ = self._correlation(x)
        else:
            corr3_ = corr3.clone()

        x = self.layer3(x)
        x3 = x.clone().detach()

        # print("corr0: {}, corr1: {}, corr2: {}".format(corr0, corr1, corr2))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, (x0, x1, x2, x3), (corr1.item(), corr2.item(), corr3.item()), (corr1_.item(), corr2_.item(), corr3_.item())


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #
    #     return x


def resneXt110_cifar(**kwargs):
    model = ResNeXt_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resneXt164_cifar(**kwargs):
    model = ResNeXt_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model

def resneXt_cifar(depth, cardinality, baseWidth, **kwargs):
    assert (depth - 2) % 9 == 0
    n = (depth - 2) / 9
    model = ResNeXt_Cifar(Bottleneck, [n, n, n], cardinality, baseWidth, **kwargs)
    return model


if __name__ == '__main__':
    net = resneXt_cifar(29, 16, 64)
    y = net(torch.randn(1, 3, 32, 32))
    print(net)
    print(y.size())
