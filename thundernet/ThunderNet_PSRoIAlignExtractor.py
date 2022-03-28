# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import BaseRoIExtractor


import math
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import importlib

_C = importlib.import_module('thundernet.PSROIAlign._C')


class _PSROIAlign(Function):
    @staticmethod
    def forward(ctx, bottom_data, bottom_rois, spatial_scale, roi_size, sampling_ratio, pooled_dim):
        ctx.spatial_scale = spatial_scale  # 1./16.
        ctx.roi_size = roi_size  # 7
        ctx.sampling_ratio = sampling_ratio  # 2
        ctx.pooled_dim = pooled_dim  # 10
        ctx.feature_size = bottom_data.size()  # (B, 490, H, W)
        num_rois = bottom_rois.size(0)  # B*K
        # (B*K, 10, 7, 7)
        top_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.float32).to(bottom_data.device)
        # (B*K, 10, 7, 7)
        argmax_data = torch.zeros([num_rois, pooled_dim, roi_size, roi_size], dtype=torch.int32).to(bottom_data.device)
        if bottom_data.is_cuda:

            _C.ps_roi_align_forward(
                bottom_data,  # (B, 490, H, W)
                bottom_rois,  # (B*K, 5), e.g. K = 128
                top_data,  # (B*K, 10, 7, 7)
                argmax_data,  # (B*K, 10, 7, 7)
                spatial_scale,  # 1./16.
                roi_size,  # 7
                sampling_ratio,  # 2
            )
            ctx.save_for_backward(bottom_rois, argmax_data)
        else:
            raise NotImplementedError

        return top_data

    @staticmethod
    @once_differentiable
    def backward(ctx, top_diff):
        spatial_scale = ctx.spatial_scale  # 1./16.
        roi_size = ctx.roi_size  # 7
        sampling_ratio = ctx.sampling_ratio  # 2
        batch_size, channels, height, width = ctx.feature_size
        [bottom_rois, argmax_data] = ctx.saved_tensors
        bottom_diff = None
        if ctx.needs_input_grad[0]:
            bottom_diff = torch.zeros([batch_size, channels, height, width], dtype=torch.float32).to(top_diff.device)
            _C.ps_roi_align_backward(
                top_diff,  # (B*K, 10, 7, 7)
                argmax_data,  # (B*K, 10, 7, 7)
                bottom_rois,  # (B*K, 10, 7, 7)
                bottom_diff,  # (B, 490, H, W)
                spatial_scale,  # 1./16.
                roi_size,  # 7
                sampling_ratio,  # 2
            )

        return bottom_diff, None, None, None, None, None


ps_roi_align = _PSROIAlign.apply


@ROI_EXTRACTORS.register_module()
class ThunderNetPSROIAlign(BaseRoIExtractor):
    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale=56, init_cfg=None):
        super(ThunderNetPSROIAlign, self).__init__(roi_layer, out_channels, featmap_strides, init_cfg)
        self.spatial_scale = 1.0 / 16.0
        self.roi_size = 7
        self.sampling_ratio = 2
        self.pooled_dim = 5

    def forward(self, bottom_data, bottom_rois):
        return ps_roi_align(
            bottom_data[0],  # (B, 490, H, W)
            bottom_rois,  # (B*K, 5)
            self.spatial_scale,  # 1./16.
            self.roi_size,  # 7
            self.sampling_ratio,  # 2
            self.pooled_dim,  # 10
        )
