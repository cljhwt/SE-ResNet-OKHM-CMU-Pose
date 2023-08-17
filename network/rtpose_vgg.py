"""CPM Pytorch Implementation"""

from collections import OrderedDict
import urllib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init
import math

#import torchvision.models as models

#resnet50 = models.resnet50(pretrained=False) #加载预训练网络

BN_MOMENTUM = 0.1
NUM_DECONV_LAYERS = 1  # 3
NUM_DECONV_FILTERS = [512]  # [256, 256, 256]
NUM_DECONV_KERNELS = [4]  # [4, 4, 4]

#mobileNet卷积模块
def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

#me write resnetXt(SE)模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #取通道上的均值
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # if reduction:
        #     self.se = SELayer(planes * 4)
        self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        # if self.reduc:
        #     out = self.se(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)#*layers将多个参数放到元组中



def get_model(trunk='vgg19'):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]



    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]
        # blocks['block%d_1' % i] = [
        #     {'Mconv1_stage%d_L1' % i: [185, 128, 3, 1, 1]},
        #     {'Mconv2_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv3_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv4_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv5_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
        #     {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        # ]
        #
        # blocks['block%d_2' % i] = [
        #     {'Mconv1_stage%d_L2' % i: [185, 128, 3, 1, 1]},
        #     {'Mconv2_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv3_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv4_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv5_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
        #     {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        # ]

    models = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models['block0'] = make_vgg19_block(block0)#产生VGG19Net

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    def get_Cpm_layer():
        Cpm_model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return Cpm_model
###-----------------SE-Resnet骨架网络, 开始-----------------###
    # class rtpose_model(nn.Module):
    #     def __init__(self, model_dict):
    #         super(rtpose_model, self).__init__()
    #         #super().__init__()
    #
    #         #self.model0 = model_dict['block0'] #VGGNet
    #
    #         #resnetXt(SE)模块
    #         self.inplanes = 64
    #         #上采样不需要偏置层
    #         self.deconv_with_bias = False
    #         #self.layers = [11, 12, 6, 3]
    #         self.layers = [3, 4, 6, 3]
    #         self.block = Bottleneck
    #         # self.conv2 = nn.Conv2d(3, 64, kernel_size=7,
    #         #                        stride=1, padding=3, bias=False)
    #         self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
    #                                stride=2, padding=3, bias=False)
    #         self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
    #         self.relu = nn.ReLU(inplace=True)
    #         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #
    #         self.layer1 = self.make_layer(self.block, 64, self.layers[0])
    #         self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
    #         self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
    #         #self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)
    #
    #         #上采样网络
    #         self.deconv_layers = self._make_deconv_layer(
    #             NUM_DECONV_LAYERS,  # 3
    #             NUM_DECONV_FILTERS,  # [256, 256, 256]
    #             NUM_DECONV_KERNELS,  # [4, 4, 4]
    #         )
    #
    #         #最后1*1网络
    #         # self.final_layer = nn.Conv2d(
    #         #     in_channels=256,
    #         #     out_channels=128,
    #         #     kernel_size=1,
    #         #     stride=1,
    #         #     padding=0
    #         # )
    #
    #         self.cpm_model = get_Cpm_layer()
    #         self.model1_1 = model_dict['block1_1']
    #         self.model2_1 = model_dict['block2_1']
    #         self.model3_1 = model_dict['block3_1']
    #         self.model4_1 = model_dict['block4_1']
    #         self.model5_1 = model_dict['block5_1']
    #         self.model6_1 = model_dict['block6_1']
    #
    #         self.model1_2 = model_dict['block1_2']
    #         self.model2_2 = model_dict['block2_2']
    #         self.model3_2 = model_dict['block3_2']
    #         self.model4_2 = model_dict['block4_2']
    #         self.model5_2 = model_dict['block5_2']
    #         self.model6_2 = model_dict['block6_2']
    #
    #         self._initialize_weights_norm()
    #
    #     def forward(self, x):
    #
    #         saved_for_loss = []
    #         #out1a = self.model0(x) #VGGNet
    #
    #         #resnetXt(SE)模块+两次上采样
    #         x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
    #         # x = self.conv1(x)
    #         # x = self.bn1(x)
    #         # x = self.relu(x)
    #         # x = x = self.maxpool(x)
    #         x = self.layer1(x)  # 256 * h/4 * w/4
    #         x = self.layer2(x)  # 512 * h/8 * w/8
    #         x = self.layer3(x)  # 1024 * h/16 * h/16
    #         #x = self.layer4(x)  # 2048 * h/32 * h/32
    #         x = self.deconv_layers(x) #256 * h/8 * h/8
    #         #out1 = self.final_layer(x) #128* h/8 * h/8
    #         out1 = self.cpm_model(x)
    #         #print(out1)
    #
    #         out1_1 = self.model1_1(out1)
    #         out1_2 = self.model1_2(out1)
    #         out2 = torch.cat([out1_1, out1_2, out1], 1)#在通道上进行合并
    #         saved_for_loss.append(out1_1)
    #         saved_for_loss.append(out1_2)
    #         #print(saved_for_loss)
    #
    #         out2_1 = self.model2_1(out2)
    #         out2_2 = self.model2_2(out2)
    #         out3 = torch.cat([out2_1, out2_2, out1], 1)
    #         saved_for_loss.append(out2_1)
    #         saved_for_loss.append(out2_2)
    #
    #         out3_1 = self.model3_1(out3)
    #         out3_2 = self.model3_2(out3)
    #         out4 = torch.cat([out3_1, out3_2, out1], 1)
    #         saved_for_loss.append(out3_1)
    #         saved_for_loss.append(out3_2)
    #
    #         out4_1 = self.model4_1(out4)
    #         out4_2 = self.model4_2(out4)
    #         out5 = torch.cat([out4_1, out4_2, out1], 1)
    #         saved_for_loss.append(out4_1)
    #         saved_for_loss.append(out4_2)
    #
    #         out5_1 = self.model5_1(out5)
    #         out5_2 = self.model5_2(out5)
    #         out6 = torch.cat([out5_1, out5_2, out1], 1)
    #         saved_for_loss.append(out5_1)
    #         saved_for_loss.append(out5_2)
    #
    #         out6_1 = self.model6_1(out6)
    #         out6_2 = self.model6_2(out6)
    #         saved_for_loss.append(out6_1)
    #         saved_for_loss.append(out6_2)
    #
    #         return (out6_1, out6_2), saved_for_loss
    #
    #
    #     # def get_Cpm_layer(self):
    #     #     Cpm_model = nn.Sequential(
    #     #         nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
    #     #         nn.ReLU(inplace=True),
    #     #         nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
    #     #         nn.ReLU(inplace=True)
    #     #     )
    #     #     return Cpm_model
    #
    #     def make_layer(self, block, planes, blocks, stride=1):
    #         downsample = None
    #         if stride != 1 or self.inplanes != planes * block.expansion:
    #             downsample = nn.Sequential(
    #                 nn.Conv2d(self.inplanes, planes * block.expansion,
    #                           kernel_size=1, stride=stride, bias=False),
    #                 nn.BatchNorm2d(planes * block.expansion),
    #             )
    #
    #         layers = []
    #         if downsample is not None:
    #             layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
    #         else:
    #             layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
    #         self.inplanes = planes * block.expansion
    #         for i in range(1, blocks):
    #             layers.append(block(self.inplanes, planes))
    #
    #
    #         #print(layers)
    #         return nn.Sequential(*layers)
    #
    #     def _get_deconv_cfg(self, deconv_kernel, index):
    #         if deconv_kernel == 4:
    #             padding = 1
    #             output_padding = 0
    #         elif deconv_kernel == 3:
    #             padding = 1
    #             output_padding = 1
    #         elif deconv_kernel == 2:
    #             padding = 0
    #             output_padding = 0
    #
    #         return deconv_kernel, padding, output_padding
    #
    #     #上采样网络
    #     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
    #         assert num_layers == len(num_filters), \
    #             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    #         assert num_layers == len(num_kernels), \
    #             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    #
    #         layers = []
    #         for i in range(num_layers):
    #             kernel, padding, output_padding = \
    #                 self._get_deconv_cfg(num_kernels[i], i)  # 4,1,0
    #
    #             planes = num_filters[i]  # 256
    #             layers.append(
    #                 nn.ConvTranspose2d(
    #                     in_channels=self.inplanes,  # 上一层输出通道
    #                     out_channels=planes,  # 256
    #                     kernel_size=kernel,  # 4
    #                     stride=2,
    #                     padding=padding,  # 1
    #                     output_padding=output_padding,  # 2
    #                     bias=self.deconv_with_bias))  # False
    #             layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
    #             layers.append(nn.ReLU(inplace=True))
    #             self.inplanes = planes
    #
    #         return nn.Sequential(*layers)
    #
    #     def _initialize_weights_norm(self):
    #
    #         # count1 = 0
    #         # for h in self.layer1[0].modules():  # 初始化nn.Sequential中类的
    #         #
    #         #     count1 = count1 + 1
    #         #     #print(h)
    #         #     #print(count)
    #         #     if count1 == 15:
    #         #         print(h[0].weight)
    #
    #         #count = 0
    #         #整体初始化,原始
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #                 #count = count + 1
    #                 #print(count)
    #                 init.normal_(m.weight, std=0.01)
    #                 if m.bias is not None:  # mobilenet conv2d doesn't add bias
    #                     init.constant_(m.bias, 0.0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #
    #         #推荐
    #         for m in self.layer1.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #                 if m.bias is not None:  # mobilenet conv2d doesn't add bias
    #                          init.constant_(m.bias, 0.0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #         for m in self.layer2.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #                 if m.bias is not None:  # mobilenet conv2d doesn't add bias
    #                          init.constant_(m.bias, 0.0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #         for m in self.layer3.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #                 if m.bias is not None:  # mobilenet conv2d doesn't add bias
    #                          init.constant_(m.bias, 0.0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 m.weight.data.fill_(1)
    #                 m.bias.data.zero_()
    #
    #         for m in self.deconv_layers.modules():
    #             if isinstance(m, nn.ConvTranspose2d):
    #                 nn.init.normal_(m.weight, std=0.001)
    #                 if self.deconv_with_bias:
    #                     nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)
    #
    #
    #         # count = 0
    #         # for h in self.layer1[0].modules():#初始化nn.Sequential中类的方法
    #         #     count = count + 1
    #         #     #print(h)
    #         #     #print(count)
    #         #     if count == 15:
    #         #         print(h[0].weight)
    #
    #         #读取对应模块权重与偏置值的方法
    #         # w1 = self.model1_1[8].weight
    #         # b1 = self.model1_1[8].bias
    #         # #d1 = self.model1_1[8].data
    #         # print(w1)
    #         # print(b1)
    #         #print(d1)
    #         #for m in self.modules():
    #
    #             #print(m)
    #
    #         # last layer of these block don't have Relu
    #         #8是因为有Relu层
    #         #单层初始化
    #         init.normal_(self.model1_1[8].weight, std=0.01)
    #         init.normal_(self.model1_2[8].weight, std=0.01)
    #
    #         init.normal_(self.model2_1[12].weight, std=0.01)
    #         init.normal_(self.model3_1[12].weight, std=0.01)
    #         init.normal_(self.model4_1[12].weight, std=0.01)
    #         init.normal_(self.model5_1[12].weight, std=0.01)
    #         init.normal_(self.model6_1[12].weight, std=0.01)
    #
    #         init.normal_(self.model2_2[12].weight, std=0.01)
    #         init.normal_(self.model3_2[12].weight, std=0.01)
    #         init.normal_(self.model4_2[12].weight, std=0.01)
    #         init.normal_(self.model5_2[12].weight, std=0.01)
    #         init.normal_(self.model6_2[12].weight, std=0.01)
    #
    #         # w1 = self.model1_1[8].weight
    #         # b1 = self.model1_1[8].bias
    #         # #d1 = self.model1_1[3].data
    #         # print(w1)
    #         # print(b1)
    #         # print(d1)
    #
    # model = rtpose_model(models)
    # return model
###-----------------结束-----------------###

###-----------------VGG19的骨架网络, 开始-----------------###
    class rtpose_model(nn.Module):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)

            # last layer of these block don't have Relu
            init.normal_(self.model1_1[8].weight, std=0.01)
            init.normal_(self.model1_2[8].weight, std=0.01)

            init.normal_(self.model2_1[12].weight, std=0.01)
            init.normal_(self.model3_1[12].weight, std=0.01)
            init.normal_(self.model4_1[12].weight, std=0.01)
            init.normal_(self.model5_1[12].weight, std=0.01)
            init.normal_(self.model6_1[12].weight, std=0.01)

            init.normal_(self.model2_2[12].weight, std=0.01)
            init.normal_(self.model3_2[12].weight, std=0.01)
            init.normal_(self.model4_2[12].weight, std=0.01)
            init.normal_(self.model5_2[12].weight, std=0.01)
            init.normal_(self.model6_2[12].weight, std=0.01)

    model = rtpose_model(models)
    return model
###-----------------结束-----------------###

###-----------------使用MobileNet以及四个stage的姿态估计网络-----------------###
def get_mobileNet_model():
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 5):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    def get_Cpm_layer():
        Cpm_model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return Cpm_model

###-----------------MobileNet的骨架网络, 开始-----------------###
    class rtpose_model(nn.Module):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.deconv_with_bias = False
            self.model0 = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512, stride=2),  # conv4_2
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )

            self.deconv_layers = self._make_deconv_layer(
                NUM_DECONV_LAYERS,  # 1
                NUM_DECONV_FILTERS,  # 512
                NUM_DECONV_KERNELS,  # 4
            )

            self.cpm_model = get_Cpm_layer()
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            x = self.model0(x)
            x = self.deconv_layers(x)
            out1 = self.cpm_model(x)
            #print(out1.shape)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)
            #print(out4_1.shape)

            return (out4_1, out4_2), saved_for_loss

        def _get_deconv_cfg(self, deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        #上采样网络
        def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
            assert num_layers == len(num_filters), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'
            assert num_layers == len(num_kernels), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)  # 4,1,0

                planes = num_filters[i]  # 512
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=512,  # 上一层输出通道
                        out_channels=planes,  # 512
                        kernel_size=kernel,  # 4
                        stride=2,
                        padding=padding,  # 1
                        output_padding=output_padding,  # 2
                        bias=self.deconv_with_bias))  # False
                layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))
                self.inplanes = planes

            return nn.Sequential(*layers)

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)

            # last layer of these block don't have Relu
            init.normal_(self.model1_1[8].weight, std=0.01)
            init.normal_(self.model1_2[8].weight, std=0.01)

            init.normal_(self.model2_1[12].weight, std=0.01)
            init.normal_(self.model3_1[12].weight, std=0.01)
            init.normal_(self.model4_1[12].weight, std=0.01)

            init.normal_(self.model2_2[12].weight, std=0.01)
            init.normal_(self.model3_2[12].weight, std=0.01)
            init.normal_(self.model4_2[12].weight, std=0.01)

    model = rtpose_model(models)
    return model
###-----------------MobileNet骨架网络结束-----------------###

###-----------------结束-----------------###

###-----------------     -----------------###

class Net_other(nn.Module):
    def __init__(self):
        super(Net_other, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

###-----------------     -----------------###


###-----------------       -----------------###

def get_model_res(trunk='vgg19'):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]



    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]
        # blocks['block%d_1' % i] = [
        #     {'Mconv1_stage%d_L1' % i: [185, 128, 3, 1, 1]},
        #     {'Mconv2_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv3_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv4_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv5_stage%d_L1' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
        #     {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        # ]
        #
        # blocks['block%d_2' % i] = [
        #     {'Mconv1_stage%d_L2' % i: [185, 128, 3, 1, 1]},
        #     {'Mconv2_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv3_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv4_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv5_stage%d_L2' % i: [128, 128, 3, 1, 1]},
        #     {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
        #     {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        # ]

    models = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models['block0'] = make_vgg19_block(block0)#产生VGG19Net

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    def get_Cpm_layer():
        Cpm_model = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        return Cpm_model
    class rtpose_model(nn.Module):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            #super().__init__()

            #self.model0 = model_dict['block0'] #VGGNet

            #resnetXt(SE)模块
            self.inplanes = 64
            #上采样不需要偏置层
            self.deconv_with_bias = False
            #self.layers = [11, 12, 6, 3]
            self.layers = [3, 4, 6, 3]
            self.block = Bottleneck
            # self.conv2 = nn.Conv2d(3, 64, kernel_size=7,
            #                        stride=1, padding=3, bias=False)
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self.make_layer(self.block, 64, self.layers[0])
            self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
            self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
            #self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

            #上采样网络
            self.deconv_layers = self._make_deconv_layer(
                NUM_DECONV_LAYERS,  # 3
                NUM_DECONV_FILTERS,  # [256, 256, 256]
                NUM_DECONV_KERNELS,  # [4, 4, 4]
            )

            #最后1*1网络
            # self.final_layer = nn.Conv2d(
            #     in_channels=256,
            #     out_channels=128,
            #     kernel_size=1,
            #     stride=1,
            #     padding=0
            # )

            self.cpm_model = get_Cpm_layer()
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            #out1a = self.model0(x) #VGGNet

            #resnetXt(SE)模块+两次上采样
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
            # x = self.conv1(x)
            # x = self.bn1(x)
            # x = self.relu(x)
            # x = x = self.maxpool(x)
            x = self.layer1(x)  # 256 * h/4 * w/4
            x = self.layer2(x)  # 512 * h/8 * w/8
            x = self.layer3(x)  # 1024 * h/16 * h/16
            #x = self.layer4(x)  # 2048 * h/32 * h/32
            x = self.deconv_layers(x) #256 * h/8 * h/8
            #out1 = self.final_layer(x) #128* h/8 * h/8
            out1 = self.cpm_model(x)
            #print(out1)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)#在通道上进行合并
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)
            #print(saved_for_loss)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss


        # def get_Cpm_layer(self):
        #     Cpm_model = nn.Sequential(
        #         nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(inplace=True)
        #     )
        #     return Cpm_model

        def make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            if downsample is not None:
                layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
            else:
                layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))


            #print(layers)
            return nn.Sequential(*layers)

        def _get_deconv_cfg(self, deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        #上采样网络
        def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
            assert num_layers == len(num_filters), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'
            assert num_layers == len(num_kernels), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'

            layers = []
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)  # 4,1,0

                planes = num_filters[i]  # 256
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=self.inplanes,  # 上一层输出通道
                        out_channels=planes,  # 256
                        kernel_size=kernel,  # 4
                        stride=2,
                        padding=padding,  # 1
                        output_padding=output_padding,  # 2
                        bias=self.deconv_with_bias))  # False
                layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
                layers.append(nn.ReLU(inplace=True))
                self.inplanes = planes

            return nn.Sequential(*layers)

        def _initialize_weights_norm(self):

            # count1 = 0
            # for h in self.layer1[0].modules():  # 初始化nn.Sequential中类的
            #
            #     count1 = count1 + 1
            #     #print(h)
            #     #print(count)
            #     if count1 == 15:
            #         print(h[0].weight)

            #count = 0
            #整体初始化,原始
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    #count = count + 1
                    #print(count)
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


            #推荐
            for m in self.layer1.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                             init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            for m in self.layer2.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                             init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            for m in self.layer3.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                             init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            for m in self.deconv_layers.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


            # count = 0
            # for h in self.layer1[0].modules():#初始化nn.Sequential中类的方法
            #     count = count + 1
            #     #print(h)
            #     #print(count)
            #     if count == 15:
            #         print(h[0].weight)

            #读取对应模块权重与偏置值的方法
            # w1 = self.model1_1[8].weight
            # b1 = self.model1_1[8].bias
            # #d1 = self.model1_1[8].data
            # print(w1)
            # print(b1)
            #print(d1)
            #for m in self.modules():

                #print(m)

            # last layer of these block don't have Relu
            #8是因为有Relu层
            #单层初始化
            init.normal_(self.model1_1[8].weight, std=0.01)
            init.normal_(self.model1_2[8].weight, std=0.01)

            init.normal_(self.model2_1[12].weight, std=0.01)
            init.normal_(self.model3_1[12].weight, std=0.01)
            init.normal_(self.model4_1[12].weight, std=0.01)
            init.normal_(self.model5_1[12].weight, std=0.01)
            init.normal_(self.model6_1[12].weight, std=0.01)

            init.normal_(self.model2_2[12].weight, std=0.01)
            init.normal_(self.model3_2[12].weight, std=0.01)
            init.normal_(self.model4_2[12].weight, std=0.01)
            init.normal_(self.model5_2[12].weight, std=0.01)
            init.normal_(self.model6_2[12].weight, std=0.01)

            # w1 = self.model1_1[8].weight
            # b1 = self.model1_1[8].bias
            # #d1 = self.model1_1[3].data
            # print(w1)
            # print(b1)
            # print(d1)

    model = rtpose_model(models)
    return model

###-----------------       -----------------###




"""Load pretrained model on Imagenet
:param model, the PyTorch nn.Module which will train.
:param model_path, the directory which load the pretrained model, will download one if not have.
:param trunk, the feature extractor network of model.               
"""

def use_vgg(model, model_path, trunk):
    model_urls = {
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'ssd': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

    number_weight = {
        'vgg16': 18,
        'ssd': 18,
        'vgg19': 20}

    url = model_urls[trunk]

    if trunk == 'ssd':
        urllib.request.urlretrieve('https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth',
                           os.path.join(model_path, 'ssd.pth')) #python2 与python3的urllib不同在与python3要加上.request,urllib.request.urlretrieve
        vgg_state_dict = torch.load(os.path.join(model_path, 'ssd.pth'))
        print('loading SSD')
    else:
        vgg_state_dict = model_zoo.load_url(url, model_dir=model_path)


    vgg_keys = vgg_state_dict.keys()
    # print(vgg_keys)
    # print(model.state_dict().keys())

    # load weights of vgg
    weights_load = {}
    # weight+bias,weight+bias.....(repeat 10 times)
    #print(list(model.state_dict().keys()))
    for i in range(number_weight[trunk]):#因为取前十层且含有weight与bias，所以为20, i为索引
        weights_load[list(model.state_dict().keys())[i]
                     ] = vgg_state_dict[list(vgg_keys)[i]]

    state = model.state_dict() #读取原始网络初始化参数
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model: {}'.format(model_path))


def use_SEResNet(model, model_path, trunk):
    model_urls = {
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'ssd': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'SEResNet': 'https://www.dropbox.com/s/xpq8ne7rwa4kg4c/seresnet50-60a8950a85b2b.pkl'}

    number_weight = {
        'vgg16': 18,
        'ssd': 18,
        'vgg19': 20,
        'SEResNet': 284 }

    vgg_state_dict = torch.load('/home/data/wangchenliang/pytorch-paf-model/ori_model/seresnet50-60a8950a85b2b.pkl')
    #vgg_keys = vgg_state_dict.keys()
    vgg_keys =  vgg_state_dict.keys()
    # for i in range(number_weight[trunk]):
    #     print(list(vgg_keys)[i])
    #     print(list(model.state_dict().keys())[i])
    # for k, v in vgg_state_dict.items():
    #     #print(k)
    #     if k == 'bn1.num_batches_tracked':
    #         print(v)
    #         print('find')

    # load weights of vgg
    weights_load = {}

    # weight+bias,weight+bias.....(repeat 10 times)
    # print(list(model.state_dict().keys()))

    for i in range(number_weight[trunk]):  # 因为取前十层且含有weight与bias，所以为20, i为索引
        weights_load[list(model.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]
        #print(vgg_state_dict[list(vgg_keys)[i]])

    # print(vgg_state_dict['conv1.weight'])
    # print(model.state_dict()['module.conv1.weight'])

    state = model.state_dict()  # 读取原始网络初始化参数
    state.update(weights_load)
    model.load_state_dict(state)
    print('load imagenet pretrained model: {}'.format(model_path))

def use_MobileNet(model, pretrained_model, trunk=16):

    model_dict = model.state_dict()

    MobileNet_state_dict_all = torch.load(pretrained_model)
    MobileNet_state_dict = MobileNet_state_dict_all['state_dict']
    MobileNet_state_dict_keys = MobileNet_state_dict.keys()

    # load weights of MobileNet
    weights_load_mobilenet = {}
    #pose-mobilenet: 242, mobilenet:
    #print('val')
    # for i in range(242):
    #     #print(list(MobileNet_state_dict)[i])
    #     print(list(model.state_dict().keys())[i])

    # for k, v in model_dict.items():
    #     print(k)
    #     print(type(k))
    #     if k == 'module.model0.1.1.num_batches_tracked':
    #         print(v)

    #print('load_weight')
    j = 0
    for i in range(138):
        if (i - 5) % 6 == 0:
            j = j + 1
        if (i - 5) % 6 != 0:
            weights_load_mobilenet[list(model_dict.keys())[i]] = MobileNet_state_dict[
                list(MobileNet_state_dict_keys)[i - j]]
            #print(list(model_dict.keys())[i])
            #print(list(MobileNet_state_dict_keys)[i - j])
            #print(weights_load_mobilenet)

    state = model.state_dict()  # 读取原始网络初始化参数
    state.update(weights_load_mobilenet)
    model.load_state_dict(state)

    #print('OK')


