# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES


class ShuffleNetV2Block(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size=3, block_idx=0):
        super(ShuffleNetV2Block, self).__init__()

        pad = kernel_size // 2

        self.block_idx = block_idx
        stride = 2 if block_idx == 0 else 1
        branch_out = out_channel - in_channel

        branch = [
            # pw
            nn.Conv2d(in_channel, mid_channel, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channel, mid_channel, kernel_size, stride, padding=pad, groups=mid_channel, bias=False),
            nn.BatchNorm2d(mid_channel),
            # pw linear
            nn.Conv2d(mid_channel, branch_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
        ]
        self.branch = nn.Sequential(*branch)
        if block_idx == 0:
            branch_left = [
                # pw
                nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding=pad, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                # dw
                nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
            ]
            self.branch_left = nn.Sequential(*branch_left)

    def forward(self, x):
        import torch

        if self.block_idx == 0:
            return torch.cat((self.branch_left(x), self.branch(x)), 1)
        else:
            x1, x2 = self.channel_shuffle(x)
            return torch.cat((x1, self.branch(x2)), 1)

    def channel_shuffle(self, x):
        batch_size, num_channel, H, W = x.data.size()
        x = x.reshape(batch_size * num_channel // 2, 2, H * W)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, batch_size, num_channel // 2, H, W)
        return x[0], x[1]


@BACKBONES.register_module()
class ShuffleNetV2(BaseModule):
    def __init__(self, init_cfg=None, pretrained=None):
        super(ShuffleNetV2, self).__init__(init_cfg=init_cfg)
        print('init_cfg:', init_cfg)
        self.init_cfg = init_cfg

        self.stage_repeats = [4, 8, 4]

        self.stage_out_channels = [-1, 24, 132, 264, 528]

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_channel = [132, 264, 528]
        stage_repeat_num = [4, 8, 4]
        in_channel = 24
        self.stage = []
        for idx_stage in range(3):
            layer = []
            out_channel = stage_channel[idx_stage]
            for idx_repeat in range(stage_repeat_num[idx_stage]):
                if idx_repeat == 0:
                    layer.append(ShuffleNetV2Block(in_channel, out_channel, out_channel // 2, 5, idx_repeat))
                else:
                    layer.append(ShuffleNetV2Block(in_channel // 2, out_channel, out_channel // 2, 5, idx_repeat))
                in_channel = out_channel
            self.stage.append(nn.Sequential(*layer))
        self.stage = nn.Sequential(*self.stage)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        # x = self.stage(x)
        ret = []
        x = self.stage[0](x)
        x = self.stage[1](x)
        ret.append(x)  # stage 3
        x = self.stage[2](x)
        ret.append(x)  # stage 4
        x = x.mean(-1, keepdim=True).mean(-2, keepdim=True)
        ret.append(x)  # glb Avg

        return ret
