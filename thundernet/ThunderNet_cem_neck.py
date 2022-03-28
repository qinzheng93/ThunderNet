# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS


@NECKS.register_module()
class ThunderNetCEM(BaseModule):
    def __init__(
        self,
        in_channels=[264, 528, 528],
        downsample_size=245,
        init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
    ):
        super(ThunderNetCEM, self).__init__(init_cfg)

        self.C4 = nn.Conv2d(in_channels[0], downsample_size, 1, bias=True)
        self.C5 = nn.Conv2d(in_channels[1], downsample_size, 1, bias=True)
        self.Cglb = nn.Conv2d(in_channels[2], downsample_size, 1, bias=True)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == 3

        C4_out = self.C4(inputs[0])
        C5_out = self.C5(inputs[1])
        C5_out = F.interpolate(C5_out, size=[C4_out.size(2), C4_out.size(3)], mode="nearest")
        x = inputs[2].mean(-1, keepdim=True).mean(-2, keepdim=True)
        Cglb_out = self.Cglb(x)  # 自动广播
        # print('shape error')
        # print(C4_out.shape)
        # print(C5_out.shape)
        # print(Cglb_out.shape)
        # exit(0)
        out = [C4_out + C5_out + Cglb_out]
        return tuple(out)
