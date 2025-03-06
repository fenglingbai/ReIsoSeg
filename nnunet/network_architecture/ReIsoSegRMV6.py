#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import torch.nn.functional
import torch.nn.functional as F
import random
import numpy as np

from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork


from timm.models.layers import trunc_normal_, DropPath

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvDropoutNormNonlin, self).__init__()

        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, bias=True)
        self.instnorm = nn.InstanceNorm3d(output_channels, eps=1e-5, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.lrelu(self.instnorm(x))

class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True},
                 first_stride=None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        self.conv_kwargs = conv_kwargs
        self.num_convs = num_convs
        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.conv_stack1 = ConvDropoutNormNonlin(self.input_channels, self.output_channels, self.conv_kwargs_first_conv)
        self.conv_stack2 = ConvDropoutNormNonlin(self.output_channels, self.output_channels, self.conv_kwargs)
    def forward(self, x):
        x = self.conv_stack1(x)
        for _ in range(self.num_convs - 1):
            x = self.conv_stack2(x)
        return x

class down(nn.Module):
    def __init__(self, input_features, output_features, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                                padding=(0, 1, 1)):
        super(down, self).__init__()
        self.conv_stack = ConvDropoutNormNonlin(input_features, output_features, 
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding)
    def forward(self, x):

        return self.conv_stack(x)


class up(nn.Module):
    def __init__(self, input_features, output_features, 
                 kernel_size=[2,2,2], stride=[2,2,2], output_padding=[0,0,0]):
        super(up, self).__init__()
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_3_t,
        # stride: _size_3_t = 1,
        # padding: _size_3_t = 0,
        # output_padding: _size_3_t = 0,

        self.up = nn.ConvTranspose3d(in_channels=input_features, 
                                     out_channels=input_features // 2, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     output_padding=output_padding)  
        self.conv_stack1 = ConvDropoutNormNonlin(input_features, input_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv_stack2 = ConvDropoutNormNonlin(input_features, output_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))

    def forward(self, x, skip):
        # tarSize = skips.shape[2:]
        # x = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
        x_up  = self.up(x)
        x_cat = torch.cat((x_up, skip), dim=1)
        x_cat = self.conv_stack1(x_cat)
        x_cat = self.conv_stack2(x_cat)
        return x_cat

class ConvT_stack_z(nn.Module):
    def __init__(self, input_features, kernel_size=[2,1,1], stride=[2,1,1], output_padding=[0,0,0]):
        super(ConvT_stack_z, self).__init__()
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_3_t,
        # stride: _size_3_t = 1,
        # padding: _size_3_t = 0,
        # output_padding: _size_3_t = 0,

        self.up = nn.ConvTranspose3d(in_channels=input_features, 
                                     out_channels=input_features, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     output_padding=output_padding)  
        self.conv_stack1 = ConvDropoutNormNonlin(input_features, input_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))

    def forward(self, x):
        x_up_z  = self.up(x)
        x_up_z = self.conv_stack1(x_up_z)
        return x_up_z


class Conv_stack_z(nn.Module):
    def __init__(self, input_features, kernel_size=[2,1,1], stride=[2,1,1]):
        super(Conv_stack_z, self).__init__()
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_3_t,
        # stride: _size_3_t = 1,
        # padding: _size_3_t = 0,
        # output_padding: _size_3_t = 0,

        self.up = nn.Conv3d(in_channels=input_features, 
                                     out_channels=input_features, 
                                     kernel_size=kernel_size, 
                                     stride=stride)  
        self.conv_stack1 = ConvDropoutNormNonlin(input_features, input_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))

    def forward(self, x):
        x_up_z  = self.up(x)
        x_up_z = self.conv_stack1(x_up_z)
        return x_up_z

class down_skip(nn.Module):
    def __init__(self, input_features, output_features, 
                 kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                                padding=(1, 1, 1)):
        super(down_skip, self).__init__()
        self.down = ConvDropoutNormNonlin(input_features, input_features * 2, 
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding)
        self.conv_stack1 = ConvDropoutNormNonlin(input_features * 4, output_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        # self.conv_stack2 = ConvDropoutNormNonlin(input_features * 2, output_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))
    def forward(self, x, skip):
        # tarSize = skips.shape[2:]
        # x = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
        x_down  = self.down(x)
        x_cat = torch.cat((x_down, skip), dim=1)
        x_cat = self.conv_stack1(x_cat)
        # x_cat = self.conv_stack2(x_cat)
        return x_cat

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class ux_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

class skip_block(nn.Module):
    def __init__(self, input_features, 
                 drop_path_rate=0., depths=2,
                 layer_scale_init_value=1e-6):
        super(skip_block, self).__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.stage = nn.Sequential(
                *[ux_block(dim=input_features, drop_path=dp_rates[j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths)])
        self.norm_layer = LayerNorm(input_features, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        # tarSize = skips.shape[2:]
        # x = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
        return self.norm_layer(self.stage(x))

class up_skip(nn.Module):
    def __init__(self, input_features, output_features, 
                 kernel_size=[2,2,2], stride=[2,2,2], output_padding=[0,0,0]):
        super(up, up_skip).__init__()
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_3_t,
        # stride: _size_3_t = 1,
        # padding: _size_3_t = 0,
        # output_padding: _size_3_t = 0,

        self.up = nn.ConvTranspose3d(in_channels=input_features, 
                                     out_channels=input_features // 2, 
                                     kernel_size=kernel_size, 
                                     stride=stride, 
                                     output_padding=output_padding)  
        self.conv_stack1 = ConvDropoutNormNonlin(input_features, input_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv_stack2 = ConvDropoutNormNonlin(input_features, output_features, kernel_size=(3, 3, 3), stride=(1, 1, 1))

    def forward(self, x, skip):
        # tarSize = skips.shape[2:]
        # x = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
        x_up  = self.up(x)
        x_cat = torch.cat((x_up, skip), dim=1)
        x_cat = self.conv_stack1(x_cat)
        x_cat = self.conv_stack2(x_cat)
        return x_cat

class resBlock(nn.Module):
    # https://github.com/torms3/Superhuman/blob/torch-0.4.0/code/rsunet.py#L145
    def __init__(self, in_planes, out_planes):
        super(resBlock, self).__init__()
        self.block1 = ConvDropoutNormNonlin(in_planes, out_planes, 
                                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.block2 = ConvDropoutNormNonlin(out_planes, out_planes, 
                                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.block3 = ConvDropoutNormNonlin(out_planes, out_planes, 
                                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.block4 = nn.InstanceNorm3d(out_planes, eps=1e-5, affine=True)
        self.block5 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block3(self.block2(residual))
        out = self.block5(self.block4(out))
        return out

class isoBlock(nn.Module):
    ##############################
    # graph
    # x2--cc11--x2l--x2iso--------------------------- PI1---------------=Cat--c4--x7iso--ct11--ct12--x7
    #                  |                                                   |
    #                  d1                                                  u2
    #                  |                                                   |
    # x3--cc21--x3l----==Cat--c1--x3iso---------------PI2----=Cat--c3--x6iso-------------------ct21--x6
    #                               |                        |
    #                               d2                       u1
    #                               |                        |
    # x4--cc31--x4l-----------------==Cat--c2--x4iso--PI3--x5iso-------------------------------ct31--x5
    ##############################
    def __init__(self, outer_features, inner_features, z_iso):
        super(isoBlock, self).__init__()
        self.outer_features = outer_features
        self.inner_features = inner_features
        self.z_iso = z_iso
        self.z_stage1 = self.z_iso[0]*self.z_iso[1]
        self.z_stage2 = self.z_iso[1]
        # extend block

        # transfer channels
        self.convc_1_1 = nn.Conv3d(in_channels=outer_features[0], 
                                   out_channels=inner_features[0], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        self.convc_2_1 = nn.Conv3d(in_channels=outer_features[1], 
                                   out_channels=inner_features[1], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        self.convc_3_1 = nn.Conv3d(in_channels=outer_features[2], 
                                   out_channels=inner_features[2], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        
        self.convc_1_2 = nn.Conv3d(in_channels=inner_features[0], 
                                   out_channels=outer_features[0], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        self.convc_2_2 = nn.Conv3d(in_channels=inner_features[1], 
                                   out_channels=outer_features[1], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        self.convc_3_2 = nn.Conv3d(in_channels=inner_features[2], 
                                   out_channels=outer_features[2], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)

        # compress block
        self.convzt_1_1 = nn.Conv3d(in_channels=inner_features[0], 
                                   out_channels=inner_features[0], 
                                   kernel_size=(z_iso[1], 1, 1), 
                                   stride=(z_iso[1], 1, 1))
        self.convzt_1_2 = nn.Conv3d(in_channels=inner_features[0], 
                                   out_channels=outer_features[0], 
                                   kernel_size=(z_iso[0], 1, 1), 
                                   stride=(z_iso[0], 1, 1))
        self.convzt_2_1 = nn.Conv3d(in_channels=inner_features[1], 
                                   out_channels=outer_features[1], 
                                   kernel_size=(z_iso[1], 1, 1), 
                                   stride=(z_iso[1], 1, 1))
        # transfer channels
        self.convzt_3_1 = nn.Conv3d(in_channels=inner_features[2], 
                                   out_channels=outer_features[2], 
                                   kernel_size=(1, 1, 1), 
                                   stride=1, padding=0)
        # up and down sample block, iso branch, iso downsample and iso upsample
        self.down1 = nn.Conv3d(in_channels=inner_features[0], 
                               out_channels=inner_features[1], 
                               kernel_size=(2,2,2), 
                               stride=(2,2,2))
        self.down2 = nn.Conv3d(in_channels=inner_features[1], 
                               out_channels=inner_features[2], 
                               kernel_size=(2,2,2), 
                               stride=(2,2,2))
        self.up1 = nn.ConvTranspose3d(in_channels=inner_features[2], 
                                      out_channels=inner_features[1], 
                                      kernel_size=(2,2,2), 
                                      stride=(2,2,2), 
                                      padding=(0, 0, 0), output_padding=(0, 0, 0))

        self.up2 = nn.ConvTranspose3d(in_channels=inner_features[1], 
                                      out_channels=inner_features[0], 
                                      kernel_size=(2,2,2), 
                                      stride=(2,2,2), 
                                      padding=(0, 0, 0), output_padding=(0, 0, 0))

        self.conv1 = resBlock(in_planes=2*inner_features[1], 
                            out_planes=inner_features[1])
        self.conv2 = resBlock(in_planes=2*inner_features[2], 
                            out_planes=inner_features[2])
        self.conv3 = resBlock(in_planes=2*inner_features[1], 
                            out_planes=inner_features[1])
        self.conv4 = resBlock(in_planes=2*inner_features[0], 
                            out_planes=inner_features[0])

        self.PI1 = skip_block(input_features=inner_features[0])
        self.PI2 = skip_block(input_features=inner_features[1])
        self.PI3 = skip_block(input_features=inner_features[2])

    def sub_iso_gen(self, x, scale, axis='x'):
        x_D, x_W, x_H = x.shape[2:]
        if axis == 'y':
            x = F.interpolate(x, size=(x_D, x_W//scale, x_H), mode='nearest')
            x = F.interpolate(x, size=(x_D * scale, x_W, x_H), mode='nearest')
        elif axis == 'x':
            x = F.interpolate(x, size=(x_D, x_W, x_H//scale), mode='nearest')
            x = F.interpolate(x, size=(x_D * scale, x_W, x_H), mode='nearest')
        elif axis == 'z':
            x = F.interpolate(x, size=(x_D * scale, x_W, x_H), mode='nearest')
        else:
            raise ValueError
        return x

    def opt_ani_gen(self, x, scale, axis='x'):
        x_D, x_W, x_H = x.shape[2:]
        if axis == 'y':
            x = F.interpolate(x, size=(x_D // scale, x_W, x_H), mode='nearest')
        elif axis == 'x':
            x = F.interpolate(x, size=(x_D // scale, x_W, x_H), mode='nearest')
        elif axis == 'z':
            x = F.interpolate(x, size=(x_D // scale, x_W, x_H), mode='nearest')
        else:
            raise ValueError
        return x

    def tensor_permute(self, x, permute_type=0):

        if permute_type == 0:
            x = x.permute(0, 1, 4, 3, 2)
        elif permute_type == 1:
            x = x.permute(0, 1, 3, 2, 4)
        elif permute_type == 2:
            x = x.permute(0, 1, 2, 4, 3)
        else:
            pass
        return x

    def forward(self, x2, x3, x4):
        X2_light = self.convc_1_1(x2)
        X3_light = self.convc_2_1(x3)
        X4_light = self.convc_3_1(x4)
        axis_sub = 'z'
        axis_rot = random.choice([0, 1, 2, 3])
        if random.random() < 0.2:
            axis_sub = random.choice(["x", "y"])
        # Lossy Interpolation
        x2_iso_sub = self.sub_iso_gen(X2_light, scale=self.z_stage1, axis=axis_sub)
        x3_iso_sub = self.sub_iso_gen(X3_light, scale=self.z_stage2 , axis=axis_sub)
        x4_iso_sub = X4_light

        x2_iso_sub = self.tensor_permute(x2_iso_sub, permute_type=axis_rot)
        x3_iso_sub = self.tensor_permute(x3_iso_sub, permute_type=axis_rot)
        x4_iso_sub = self.tensor_permute(x4_iso_sub, permute_type=axis_rot)

        x3iso = self.conv1(torch.cat((x3_iso_sub, self.down1(x2_iso_sub)), dim=1))
        x4iso = self.conv2(torch.cat((x4_iso_sub, self.down2(x3iso)), dim=1))
        x5iso = self.PI3(x4iso)
        x6iso = self.conv3(torch.cat((self.up1(x5iso), self.PI2(x3iso)), dim=1))
        x7iso = self.conv4(torch.cat((self.up2(x6iso), self.PI1(x2_iso_sub)), dim=1))
        
        x5iso = self.tensor_permute(x5iso, permute_type=axis_rot)
        x6iso = self.tensor_permute(x6iso, permute_type=axis_rot)
        x7iso = self.tensor_permute(x7iso, permute_type=axis_rot)

        x5_light = x5iso
        x6_light = self.opt_ani_gen(x6iso, scale=self.z_stage2, axis=axis_sub)
        x7_light = self.opt_ani_gen(x7iso, scale=self.z_stage1, axis=axis_sub)

        x5 = self.convc_3_2(x5_light)
        x6 = self.convc_2_2(x6_light)
        x7 = self.convc_1_2(x7_light)
        return x5, x6, x7

class ReIsoSeg(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, 
                        num_features=[28, 36, 48, 64, 80], 
                        num_classes=2, 
                        ani_scale=[1,1,1],
                        deep_supervision=True):

        super(ReIsoSeg, self).__init__()
        # ----------------------------------------parameters of NNUNet
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = deep_supervision
        self.num_classes = num_classes
        # ----------------------------------------
        # compute downsample size:
        # con: out_size = (in_size + 2P - K + 1) / S
        # conT: out_size = (in_size - 1) * S + K - 2P + output_padding
        down_size_list = []
        for ani_scale_item in ani_scale:
            assert ani_scale_item > 0
            if ani_scale_item == 2:
                down_size_list.append((1, ani_scale_item, ani_scale_item))
            elif ani_scale_item == 1:
                down_size_list.append((2, 2, 2))
            else:
                raise ValueError('ani_scale must be 1 or 2')

        #############################
        self.conv0 = ConvDropoutNormNonlin(input_channels, num_features[0], 
                                           kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.conv1 = resBlock(in_planes=num_features[0], out_planes=num_features[0])
        self.down1 = nn.MaxPool3d(kernel_size=down_size_list[0], stride=down_size_list[0])

        self.conv2 = resBlock(in_planes=num_features[0], out_planes=num_features[1])
        self.down2 = nn.MaxPool3d(kernel_size=down_size_list[1], stride=down_size_list[1])

        self.conv3 = resBlock(in_planes=num_features[1], out_planes=num_features[2])
        self.down3 = nn.MaxPool3d(kernel_size=down_size_list[2], stride=down_size_list[2])

        self.conv4 = resBlock(in_planes=num_features[2], out_planes=num_features[3])
        self.down4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))


        self.center = resBlock(in_planes=num_features[3], out_planes=num_features[4])
        # out_size = (in_size - 1) * S + K - 2P + output_padding
        self.up1 = nn.ConvTranspose3d(num_features[4], num_features[3], 
                                    kernel_size=(2, 2, 2), stride=(2, 2, 2), 
                                    padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv5 = resBlock(in_planes=num_features[3]*2, out_planes=num_features[3])

        self.up2 = nn.ConvTranspose3d(num_features[3], num_features[2], 
                                    kernel_size=down_size_list[2], stride=down_size_list[2], 
                                    padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv6 = resBlock(in_planes=num_features[2]*2, out_planes=num_features[2])

        self.up3 = nn.ConvTranspose3d(num_features[2], num_features[1], 
                                    kernel_size=down_size_list[1], stride=down_size_list[1], 
                                    padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv7 = resBlock(in_planes=num_features[1]*2, out_planes=num_features[1])

        self.up4 = nn.ConvTranspose3d(num_features[1], num_features[0], 
                                    kernel_size=down_size_list[0], stride=down_size_list[0], 
                                    padding=(0, 0, 0), output_padding=(0, 0, 0))
        self.conv8 = resBlock(in_planes=num_features[0]*2, out_planes=num_features[0])

        self.conv9 = ConvDropoutNormNonlin(num_features[0], num_features[0], 
                                           kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))

        self.isoBlock = isoBlock(outer_features=num_features[1:4], 
                        inner_features=[num_features[1]//4, num_features[2]//4, num_features[3]//4], 
                        z_iso=ani_scale[1:3])

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
             for c in [num_features[0], num_features[0], num_features[1], num_features[0], num_features[0], num_features[1]]]
        )
        #############################

        self.apply(InitWeights_He(1e-2))
            # self.apply(print_module_training_status)

    def forward(self, x):
        ######################
        # ani branch
        x1 = self.conv1(self.conv0(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv3(self.down2(x2))
        x4 = self.conv4(self.down3(x3))
        x5 = self.center(self.down4(x4))

        x6 = self.conv5(torch.cat((self.up1(x5), x4), dim=1))
        x7 = self.conv6(torch.cat((self.up2(x6), x3), dim=1))
        x8 = self.conv7(torch.cat((self.up3(x7), x2), dim=1))
        x9 = self.conv8(torch.cat((self.up4(x8), x1), dim=1))
        x10 = self.conv9(x9)
        ######################
        # iso branch
        x5_skip, x6_skip, x7_skip = self.isoBlock(x2, x3, x4)
        x6_iso = self.conv5(torch.cat((self.up1(x5), x5_skip), dim=1))
        x7_iso = self.conv6(torch.cat((self.up2(x6_iso), x6_skip), dim=1))
        x8_iso = self.conv7(torch.cat((self.up3(x7_iso), x7_skip), dim=1))
        x9_iso = self.conv8(torch.cat((self.up4(x8_iso), x1), dim=1))
        x10_iso = self.conv9(x9_iso)
        ######################## !!!!!!
        if self._deep_supervision and self.do_ds:
            features = [x10_iso, x9_iso, x8_iso, x10, x9, x8]
            return tuple([self.outputs[i](features[i]) for i in range(6)])
            # return  tuple([self.conv_class[0](x9)])
        else:
            # return self.outputs[0](x10_iso)
            return self.outputs[3](x10)

if __name__ == '__main__':
    # image_size = (16, 320, 320)
    image_size = (16, 320, 320)
    # image_size = (16, 256, 256)
    ReIsoSeg = ReIsoSeg(input_channels=1, ani_scale=[1,2,2],)
    # input = torch.randn((1, 1, 19, 255, 256))
    input = torch.randn((1, 1, image_size[0], image_size[1], image_size[2]))
    output = ReIsoSeg(input)

    print([e.shape for e in output])
    from thop import profile
    flops, params = profile(ReIsoSeg, inputs=(input, ))
    print('FLOPs (G): %.2f, Params (M): %.2f'%(flops/1e9, params/1e6)) 
    print('ok')
