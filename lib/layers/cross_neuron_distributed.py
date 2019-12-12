# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.linalg import block_diag

class _CrossNeuronBlock(nn.Module):
    def __init__(self, in_channels, in_height, in_width,
                    nblocks_channel=8,
                    spatial_height=32, spatial_width=32,
                    reduction=4, size_is_consistant=True):
        # nblock_channel: number of block along channel axis
        # spatial_size: spatial_size
        super(_CrossNeuronBlock, self).__init__()

        self.corr_bf = 0
        self.corr_af = 0

        self.nblocks_channel = 1 if in_channels <= 512 else in_channels // 512

        block_size = in_channels // self.nblocks_channel
        block = torch.Tensor(block_size, block_size).fill_(1)
        self.mask = torch.Tensor(in_channels, in_channels).fill_(0)
        for i in range(self.nblocks_channel):
            self.mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size].copy_(block)

        factor = in_height // spatial_height
        if factor == 0 and size_is_consistant:
            self.spatial_area = in_height * in_width
            self.spatial_height = in_height
            self.spatial_width = in_width
        else:
            ds_layers = []
            us_layers = []
            for i in range(factor - 1):
                ds_layer = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True),
                )
                ds_layers.append(ds_layer)

                us_layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True),
                )
                us_layers.append(us_layer)
            self.downsample = nn.Sequential(*ds_layers)
            self.upsample = nn.Sequential(*us_layers)
            self.spatial_height = in_height // factor
            self.spatial_width = in_width // factor
            self.spatial_area = self.spatial_height * self.spatial_width

        self.fc_in = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            # nn.BatchNorm1d(self.spatial_area // reduction),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
        )

        self.fc_out = nn.Sequential(
            nn.Conv1d(self.spatial_area, self.spatial_area // reduction, 1, 1, 0, bias=True),
            # nn.BatchNorm1d(self.spatial_area // reduction),
            nn.ReLU(True),
            nn.Conv1d(self.spatial_area // reduction, self.spatial_area, 1, 1, 0, bias=True),
            # nn.BatchNorm1d(self.spatial_area)
        )

        self.ln = nn.LayerNorm(in_channels)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def _compute_correlation(self, x):
    #     bt, c, h, w = x.shape
    #     x_v = x.view(bt, c, h*w).detach()
    #     x_v_mean = x_v.mean(1).unsqueeze(1)
    #     x_v_cent = x_v - x_v_mean # bt x c x (hw)
    #     # x_v_cent = x_v
    #     x_v_cent = x_v_cent / (torch.norm(x_v_cent, 2, 2).unsqueeze(2) + 1e-5)
    #     correlations = torch.bmm(x_v_cent, x_v_cent.permute(0, 2, 1).contiguous()) # btxcxc
    #     diags = 1 - torch.eye(c).unsqueeze(0).type_as(correlations)
    #     correlations = correlations * diags
    #     return torch.abs(correlations).mean(0).sum() / c / (c - 1)

    def _compute_correlation(self, x):
        b, c, h, w = x.shape
        x_v = x.clone().detach().view(b, c, -1) # b x c x (hw)
        x_m = x_v.mean(1).unsqueeze(1) # b x 1 x (hw)
        x_c = x_v - x_m # b x c x (hw)
        num = torch.bmm(x_c, x_c.transpose(1, 2)) # b x c x c
        x_mode = torch.sqrt(torch.sum(x_c ** 2, 2).unsqueeze(2)) # b x c x 1
        dec = torch.bmm(x_mode, x_mode.transpose(1, 2).contiguous()) # b x c x c
        out = num / dec # b x c x c
        # diags = 1 - torch.eye(c).unsqueeze(0).type_as(out)
        # out = out * diags
        out = torch.abs(out) # .mean(0).sum() / c / (c - 1)
        return out.mean()

    def forward(self, x):
        '''
        :param x: (bt, c, h, w)
        :return:
        '''
        bt, c, h, w = x.shape
        residual = x
        x_stretch = x.view(bt, c, h * w)
        self.corr_bf = self._compute_correlation(x)
        # self.corr_af = self._compute_correlation(x)
        # return x

        if self.spatial_height == h and self.spatial_width == w:
            x_stacked = x_stretch # (b) x c x (h * w)
            x_stacked = x_stacked.view(bt * self.nblocks_channel, c // self.nblocks_channel, -1)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c
            x_v = self.fc_in(x_v) # (b) x (h * w) x c
            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel) # (b * h * w) x 1 x c
            # x_m = x_m - x_m.mean(2).unsqueeze(2)
            # x_m = self.ln(x_m)
            # x_m = x_m.detach()
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * h * w) x c x c
            # score = -score / score.sum(2).unsqueeze(2)
            # score = torch.bmm(x_v.transpose(1, 2).contiguous(), x_v)
            # x_v = F.dropout(x_v, 0.1, self.training)
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c
            out = self.fc_out(torch.bmm(x_v, attn)) # (b) x (h * w) x c
            out = F.dropout(out.permute(0, 2, 1).contiguous().view(bt, c, h, w), 0.0, self.training)
            out = F.relu(residual + out)
            self.corr_af = self._compute_correlation(out)
            return out
        else:
            # x = self.downsample(x)
            x = F.interpolate(x, (self.spatial_height, self.spatial_width))
            x_stretch = x.view(bt, c, self.spatial_height * self.spatial_width)
            x_stretch = x.view(bt * self.nblocks_channel, c // self.nblocks_channel, self.spatial_height * self.spatial_width)

            x_stacked = x_stretch # (b) x c x (h * w)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c
            x_v = self.fc_in(x_v) # (b) x (h * w) x c
            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel) # b x 1 x c
            # x_m = x_m - x_m.mean(2).unsqueeze(2)
            # x_m = self.ln(x_m)
            # x_m = x_m.detach()
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # b x c x c
            # score = -score / score.sum(2).unsqueeze(2)
            # score = torch.bmm(x_v.transpose(1, 2).contiguous(), x_v)
            # x_v = F.dropout(x_v, 0.1, self.training)
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c
            out = self.fc_out(torch.bmm(x_v, attn)) # (b) x (h * w) x c
            out = out.permute(0, 2, 1).contiguous().view(bt, c, self.spatial_height, self.spatial_width)
            # out = F.dropout(self.upsample(out), 0.0, self.training)
            out = F.dropout(F.interpolate(out, (h, w)), 0.0, self.training)
            out = F.relu(residual + out)
            self.corr_af = self._compute_correlation(out)
            return out

# class _CrossNeuronBlock(nn.Module):
#     def __init__(self, in_channels, in_height, in_width,
#                     nblocks_channel=8,
#                     spatial_height=32, spatial_width=32,
#                     reduction=4, size_is_consistant=True):
#         # nblock_channel: number of block along channel axis
#         # spatial_size: spatial_size
#         super(_CrossNeuronBlock, self).__init__()
#
#         # set channel splits
#         if in_channels <= 512:
#             self.nblocks_channel = 1
#         else:
#             self.nblocks_channel = in_channels // 512
#         block_size = in_channels // self.nblocks_channel
#         block = torch.Tensor(block_size, block_size).fill_(1)
#         self.mask = torch.Tensor(in_channels, in_channels).fill_(0)
#         for i in range(self.nblocks_channel):
#             self.mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size].copy_(block)
#
#         # set spatial splits
#         if in_height * in_width < 32 * 32 and size_is_consistant:
#             self.spatial_area = in_height * in_width
#             self.spatial_height = in_height
#             self.spatial_width = in_width
#         else:
#             self.spatial_area = spatial_height * spatial_width
#             self.spatial_height = spatial_height
#             self.spatial_width = spatial_width
#
#         self.conv_in = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#         )
#
#         self.conv_out = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#         )
#
#         # self.bn = nn.BatchNorm1d(self.spatial_area)
#
#         self.initialize()
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         '''
#         :param x: (bt, c, h, w)
#         :return:
#         '''
#         # import pdb; pdb.set_trace()
#         bt, c, h, w = x.shape
#         residual = x
#         x_v = self.conv_in(x) # b x c x h x w
#         x_m = x_v.mean(3).mean(2).unsqueeze(2) # bt x c x 1
#
#         score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # bt x c x c
#         # score = torch.bmm(x_v.transpose(1, 2).contiguous(), x_v)
#         # x_v = F.dropout(x_v, 0.1, self.training)
#         # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
#         attn = F.softmax(score, dim=2) # bt x c x c
#         out = self.conv_out(torch.bmm(attn, x_v.view(bt, c, h * w)).view(bt, c, h, w))
#         return F.relu(residual + out)

class CrossNeuronlBlock2D(_CrossNeuronBlock):
    def __init__(self, in_channels, in_height, in_width, spatial_height, spatial_width, reduction=8, size_is_consistant=True):
        super(CrossNeuronlBlock2D, self).__init__(in_channels, in_height, in_width,
                                              nblocks_channel=4,
                                              spatial_height=spatial_height,
                                              spatial_width=spatial_width,
                                              reduction=reduction,
                                              size_is_consistant=size_is_consistant)


class CrossNeuronWrapper(nn.Module):
    def __init__(self, block, in_channels, in_height, in_width, spatial_height, spatial_width, reduction=8):
        super(CrossNeuronWrapper, self).__init__()
        self.block = block
        self.cn = CrossNeuronlBlock2D(in_channels, in_height, in_width, spatial_height, spatial_width, reduction=reduction)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, 4 * in_channels, 3, 1, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(4 * in_channels, in_channels, 3, 1, 1),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        #     )

    def forward(self, x):
        x = self.cn(x)
        x = self.block(x)
        return x

def add_cross_neuron(net, img_height, img_width, spatial_height, spatial_width, reduction=[4,4,4,4]):
    import torchvision
    import lib.networks as archs

    import pdb; pdb.set_trace()

    if isinstance(net, torchvision.models.ResNet):
        dummy_img = torch.randn(1, 3, img_height, img_width)
        out = net.conv1(dummy_img)
        out = net.relu(net.bn1(out))
        out0 = net.maxpool(out)
        print("layer0 out shape: {}x{}x{}x{}".format(out0.shape[0], out0.shape[1], out0.shape[2], out0.shape[3]))
        out1 = net.layer1(out0)
        print("layer1 out shape: {}x{}x{}x{}".format(out1.shape[0], out1.shape[1], out1.shape[2], out1.shape[3]))
        out2 = net.layer2(out1)
        print("layer2 out shape: {}x{}x{}x{}".format(out2.shape[0], out2.shape[1], out2.shape[2], out2.shape[3]))
        out3 = net.layer3(out2)
        print("layer3 out shape: {}x{}x{}x{}".format(out3.shape[0], out3.shape[1], out3.shape[2], out3.shape[3]))
        out4 = net.layer4(out3)
        print("layer4 out shape: {}x{}x{}x{}".format(out4.shape[0], out4.shape[1], out4.shape[2], out4.shape[3]))


        layers = []
        l = len(net.layer1)
        for i in range(l):
            if i == 0:
                layers.append(CrossNeuronWrapper(net.layer1[i], out0.shape[1], out0.shape[2], out0.shape[3],
                    out0.shape[2], out0.shape[3], 4))
            elif i % 4 == 0:
                layers.append(CrossNeuronWrapper(net.layer1[i], out1.shape[1], out1.shape[2], out1.shape[3],
                    out1.shape[2], out1.shape[3], 4))
            else:
                layers.append(net.layer1[i])
        net.layer1 = nn.Sequential(*layers)

        layers = []
        l = len(net.layer2)
        for i in range(l):
            if i == 0:
                layers.append(CrossNeuronWrapper(net.layer2[i], out1.shape[1], out1.shape[2], out1.shape[3],
                    out1.shape[2], out1.shape[3], 4))
            elif i % 4 == 0:
                layers.append(CrossNeuronWrapper(net.layer2[i], out2.shape[1], out2.shape[2], out2.shape[3],
                    out2.shape[2], out2.shape[3], 4))
            else:
                layers.append(net.layer2[i])
        net.layer2 = nn.Sequential(*layers)

        #
        layers = []
        l = len(net.layer3)
        for i in range(0, l):
            if i == 0:
                layers.append(CrossNeuronWrapper(net.layer3[i], out2.shape[1], out2.shape[2], out2.shape[3],
                    out2.shape[2], out2.shape[3], 4))
            elif i % 4 == 0:
                layers.append(CrossNeuronWrapper(net.layer3[i], out3.shape[1], out3.shape[2], out3.shape[3],
                    out3.shape[2], out3.shape[3], 4))
            else:
                layers.append(net.layer3[i])
        net.layer3 = nn.Sequential(*layers)

        layers = []
        l = len(net.layer4)
        for i in range(0, l):
            if i == 0:
                layers.append(CrossNeuronWrapper(net.layer4[i], out3.shape[1], out3.shape[2],  out3.shape[3],
                    out3.shape[2],  out3.shape[3], 4))
            else:
                layers.append(net.layer4[i])
        net.layer4 = nn.Sequential(*layers)

    else:
        dummy_img = torch.randn(1, 3, img_height, img_width)
        out = net.conv1(dummy_img)
        out0 = net.relu(net.bn1(out))
        out1 = net.layer1(out0)
        out2 = net.layer2(out1)
        out3 = net.layer3(out2)
        #
        net.layer1 = CrossNeuronWrapper(net.layer1, out0.shape[1], out0.shape[2], out0.shape[3], spatial_height[0], spatial_width[0])
        net.layer2 = CrossNeuronWrapper(net.layer2, out1.shape[1], out1.shape[2], out1.shape[3], spatial_height[1], spatial_width[1])
        net.layer3 = CrossNeuronWrapper(net.layer3, out2.shape[1], out2.shape[2], out2.shape[3], spatial_height[2], spatial_width[2])

        '''
        layers = []
        l = len(net.layer1)
        for i in range(l):
            if i == 0:
                layers.append(CrossNeuronWrapper(net.layer1[i], out0.shape[1], out0.shape[2],  out0.shape[3], out0.shape[2],  out0.shape[3]))
            elif i in  [4, 7]: # resnet56: [4, 7]
                layers.append(CrossNeuronWrapper(net.layer1[i], out1.shape[1], out1.shape[2],  out1.shape[3], out1.shape[2],  out1.shape[3]))
            else:
                layers.append(net.layer1[i])
        net.layer1 = nn.Sequential(*layers)
        #
        layers = []
        l = len(net.layer2)
        for i in range(l):
            if i in [0]:
                layers.append(CrossNeuronWrapper(net.layer2[i], out1.shape[1], out1.shape[2], out1.shape[3], out1.shape[2], out1.shape[3]))
            elif i in [4, 7]:
                layers.append(CrossNeuronWrapper(net.layer2[i], out2.shape[1], out2.shape[2], out2.shape[3], out2.shape[2], out2.shape[3]))
            else:
                layers.append(net.layer2[i])
        net.layer2 = nn.Sequential(*layers)
        #
        layers = []
        l = len(net.layer3)
        for i in range(0, l):
            if i in [0]:
                layers.append(CrossNeuronWrapper(net.layer3[i], out2.shape[1], out2.shape[2], out2.shape[3], out2.shape[2], out2.shape[3]))
            else:
                layers.append(net.layer3[i])
        net.layer3 = nn.Sequential(*layers)
        #
        '''
    # else:
    #     raise NotImplementedError

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    img = torch.randn(2, 3, 10, 20, 20)
    net = CrossNeuronlBlock3D(3, 20 * 20)
    out = net(img)
    print(out.size())
