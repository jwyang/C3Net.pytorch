import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.net1 = nn.Conv1d(dim, dim, 1, 1, 0)
        self.net2 = nn.Conv1d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x1 = x[0].view(x[0].shape[0], x[0].shape[1], -1)
        x2 = x[1].view(x[1].shape[0], x[1].shape[1], -1)
        x1 = F.relu(self.net1(x1)) # B x C x M
        x2 = F.relu(self.net2(x2)) # B x C x 1

        attn = torch.softmax((x1 * x2).sum(1), dim=1)
        attn = attn.unsqueeze(2)

        out = (x2 + torch.bmm(x1, attn)).unsqueeze(3)
        return out

class INVCNN(nn.Module):
    def __init__(self, nlayers, num_classes=100, has_gtlayer=False, has_selayer=False):
        super(INVCNN, self).__init__()
        self.name = "invcnn"
        self.planes = 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(64, num_classes)

        self.layer2 = self._make_layer(BasicBlock, 64, 1, kernel_size=4)
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            # nn.ReLU(True),
            )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
            )
        self.attn2 = attention(64)
        self.fc2 = nn.Linear(64, num_classes)

        self.layer3 = self._make_layer(BasicBlock, 64, 2, kernel_size=4)
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            # nn.ReLU(True),
            )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
            )
        self.attn3 = attention(64)
        self.fc3 = nn.Linear(64, num_classes)

        self.layer4 = self._make_layer(BasicBlock, 64, 3, kernel_size=4)
        self.conv1x1_4 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            # nn.ReLU(True),
            )
        self.conv3x3_4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
            )
        self.attn4 = attention(64)
        self.fc4 = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=3):
        layers = []
        inplanes = 3
        for i in range(0, blocks):
            if i == 0:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=kernel_size, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(planes)
                )
                layers.append(block(inplanes, planes, kernel_size=kernel_size, stride=kernel_size, padding=kernel_size // 2, downsample=downsample))
            else:
                layers.append(block(inplanes, planes))
            inplanes = planes
        # layers.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # out1 = self.layer1(x)
        # score1 = self.fc1(out1.view(out1.size(0), -1))
        # out2 = self.layer2(x)
        # out2 = self.conv1x1_2(out2) + F.interpolate(out1, (out2.shape[2], out2.shape[3]))
        # out2_fc = self.conv3x3_2(out2)
        # score2 = self.fc2(out2_fc.view(out2_fc.size(0), -1))
        # out3 = self.layer3(x)
        # out3 = self.conv1x1_3(out3) + F.interpolate(out2, (out3.shape[2], out3.shape[3]))
        # out3_fc = self.conv3x3_3(out3)
        # score3 = self.fc3(out3_fc.view(out3_fc.size(0), -1))
        # out4 = self.layer4(x)
        # out4 = self.conv1x1_4(out4) + F.interpolate(out3, (out4.shape[2], out4.shape[3]))
        # out4_fc = self.conv3x3_4(out4)
        # score4 = self.fc4(out4_fc.view(out4_fc.size(0), -1))
        # # scores = (score1, score2, score3, score4)
        # return [score4]

        out1 = self.layer1(F.interpolate(x, (4, 4)))
        score1 = self.fc1(out1.view(out1.size(0), -1))
        out2 = self.layer2(F.interpolate(x, (8, 8))) # BxCx2x2
        out2 = self.attn2((out2, out1))
        score2 = self.fc2(out2.view(out2.size(0), -1))
        out3 = self.layer3(F.interpolate(x, (16, 16)))
        out3 = self.attn3((out3, out2))
        score3 = self.fc3(out3.view(out3.size(0), -1))
        out4 = self.layer4(x)
        out4 = self.attn4((out4, out3))
        score4 = self.fc4(out4.view(out4.size(0), -1))
        # scores = (score1, score2, score3, score4)
        return [score4]

def inv_cnn_4(**kwargs):
    return INVCNN(4, **kwargs)
