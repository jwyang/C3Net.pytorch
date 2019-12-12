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
                    nblocks_channel=4,
                    spatial_height=24, spatial_width=24,
                    reduction=8, size_is_consistant=True):
        # nblock_channel: number of block along channel axis
        # spatial_size: spatial_size
        super(_CrossNeuronBlock, self).__init__()

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
        if in_height * in_width < 32 * 32 and size_is_consistant:
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

        import pdb; pdb.set_trace()
        
        if spblock_h == 1 and spblock_w == 1:
            x_stacked = x_stretch # (b) x c x (h * w)
            x_stacked = x_stacked.view(bt * self.nblocks_channel, c // self.nblocks_channel, -1)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b) x (h * w) x c
            x_v = self.fc_in(x_v) # (b) x (h * w) x c
            x_m = x_v.mean(1).view(-1, 1, c // self.nblocks_channel).detach() # (b * h * w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * h * w) x c x c
            # score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * h * w) x c x c
            out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b) x (h * w) x c
            out = out.permute(0, 2, 1).contiguous().view(bt, c, h, w)
            return F.relu(residual + out)
        else:
            # first splt input tensor into chunks
            ind_chunks = []
            x_chunks = []
            for i in range(spblock_h):
                for j in range(spblock_w):
                    tl_y, tl_x = max(0, i * stride_h), max(0, j * stride_w)
                    br_y, br_x = min(h, tl_y + self.spatial_height), min(w, tl_x + self.spatial_width)
                    ind_y = torch.arange(tl_y, br_y).view(-1, 1)
                    ind_x = torch.arange(tl_x, br_x).view(1, -1)
                    ind = (ind_y * w + ind_x).view(1, 1, -1).repeat(bt, c, 1).type_as(x_stretch).long()
                    ind_chunks.append(ind)
                    chunk_ij = torch.gather(x_stretch, 2, ind).contiguous()
                    x_chunks.append(chunk_ij)

            x_stacked = torch.cat(x_chunks, 0) # (b * nb_h * n_w) x c x (b_h * b_w)
            x_v = x_stacked.permute(0, 2, 1).contiguous() # (b * nb_h * n_w) x (b_h * b_w) x c
            x_v = self.fc_in(x_v) # (b * nb_h * n_w) x (b_h * b_w) x c
            x_m = x_v.mean(1).view(-1, 1, c) # (b * nb_h * n_w) x 1 x c
            score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # (b * nb_h * n_w) x c x c
            score.masked_fill_(self.mask.unsqueeze(0).expand_as(score).type_as(score).eq(0), -np.inf)
            attn = F.softmax(score, dim=1) # (b * nb_h * n_w) x c x c
            out = self.bn(self.fc_out(torch.bmm(x_v, attn))) # (b * nb_h * n_w) x (b_h * b_w) x c

            # put back to original shape
            out = out.permute(0, 2, 1).contiguous() # (b * nb_h * n_w)  x c x (b_h * b_w)
            # x_stretch_out = x_stretch.clone().zero_()
            for i in range(spblock_h):
                for j in range(spblock_w):
                    idx = i * spblock_w + j
                    ind = ind_chunks[idx]
                    chunk_ij = out[idx * bt:(idx+1) * bt]
                    x_stretch = x_stretch.scatter_add(2, ind, chunk_ij / spblock_h / spblock_h)
            return F.relu(x_stretch.view(residual.shape))

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

    def forward(self, x):
        x = self.cn(x)
        x = self.block(x)
        return x

def add_cross_neuron(net, img_height, img_width, spatial_height, spatial_width, reduction=8):
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

        # net.layer1 = CrossNeuronWrapper(net.layer1, out1.shape[1], out1.shape[2], out1.shape[3], spatial_height[0], spatial_width[0], reduction)
        net.layer2 = CrossNeuronWrapper(net.layer2, out2.shape[1], out2.shape[2], out2.shape[3], spatial_height[1], spatial_width[1], reduction)
        net.layer3 = CrossNeuronWrapper(net.layer3, out3.shape[1], out3.shape[2], out3.shape[3], spatial_height[2], spatial_width[2], reduction)
        net.layer4 = CrossNeuronWrapper(net.layer4, out4.shape[1], out4.shape[2], out4.shape[3], spatial_height[3], spatial_width[3], reduction)

        # layers = []
        # l = len(net.layer2)
        # for i in range(l):
        #     if i % 6 == 0 or i == (l - 1):
        #         layers.append(CrossNeuronWrapper(net.layer2[i], out2.shape[1], out2.shape[2], out2.shape[3],
        #             spatial_height[1], spatial_width[1], reduction[1]))
        #     else:
        #         layers.append(net.layer2[i])
        # net.layer2 = nn.Sequential(*layers)
        #
        # #
        # layers = []
        # l = len(net.layer3)
        # for i in range(0, l):
        #     if i % 6 == 0 or i == (l - 1):
        #         layers.append(CrossNeuronWrapper(net.layer3[i], out3.shape[1], out3.shape[2], out3.shape[3],
        #             spatial_height[2], spatial_width[2], reduction[2]))
        #     else:
        #         layers.append(net.layer3[i])
        # net.layer3 = nn.Sequential(*layers)
        #
        # layers = []
        # l = len(net.layer4)
        # for i in range(0, l):
        #     if i % 6 == 0 or i == (l - 1):
        #         layers.append(CrossNeuronWrapper(net.layer4[i], out4.shape[1], out4.shape[2],  out4.shape[3],
        #             spatial_height[3], spatial_width[3], reduction[3]))
        #     else:
        #         layers.append(net.layer4[i])
        # net.layer4 = nn.Sequential(*layers)


    elif isinstance(net, archs.resnet_cifar.ResNet_Cifar):

        dummy_img = torch.randn(1, 3, img_height, img_width)
        out = net.conv1(dummy_img)
        out0 = net.relu(net.bn1(out))
        out1 = net.layer1(out0)
        out2 = net.layer2(out1)
        out3 = net.layer3(out2)

        net.layer1 = CrossNeuronWrapper(net.layer1, out0.shape[1], out0.shape[2], out0.shape[3], spatial_height[0], spatial_width[0])
        net.layer2 = CrossNeuronWrapper(net.layer2, out1.shape[1], out1.shape[2], out1.shape[3], spatial_height[1], spatial_width[1])
        net.layer3 = CrossNeuronWrapper(net.layer3, out2.shape[1], out2.shape[2], out2.shape[3], spatial_height[2], spatial_width[2])

    else:
        dummy_img = torch.randn(1, 3, img_height, img_width)
        out = net.conv1(dummy_img)
        out = net.relu(net.bn1(out))
        out1 = net.layer1(out)
        out2 = net.layer2(out1)
        out3 = net.layer3(out2)

        net.layer1 = CrossNeuronWrapper(net.layer1, out1.shape[1], out1.shape[2], out1.shape[3], spatial_height[0], spatial_width[0])
        net.layer2 = CrossNeuronWrapper(net.layer2, out2.shape[1], out2.shape[2], out2.shape[3], spatial_height[1], spatial_width[1])
        net.layer3 = CrossNeuronWrapper(net.layer3, out3.shape[1], out3.shape[2], out3.shape[3], spatial_height[2], spatial_width[2])


        # layers = []
        # l = len(net.layer2)
        # for i in range(l):
        #     if i % 5 == 0 or i == (l - 1):
        #         layers.append(CrossNeuronWrapper(net.layer2[i], out2.shape[1], out2.shape[2] * out2.shape[3]))
        #     else:
        #         layers.append(net.layer2[i])
        # net.layer2 = nn.Sequential(*layers)
        # #
        # layers = []
        # l = len(net.layer3)
        # for i in range(0, l):
        #     if i % 5 == 0 or i == (l - 1):
        #         layers.append(CrossNeuronWrapper(net.layer3[i], out3.shape[1], out3.shape[2] * out3.shape[3]))
        #     else:
        #         layers.append(net.layer3[i])
        # net.layer3 = nn.Sequential(*layers)
        #
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
