# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from .ThunderNet_detector import ThunderNetDetector


@DETECTORS.register_module()
class ThunderNet(ThunderNetDetector):
    def __init__(self, backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=None, pretrained=None, init_cfg=None):
        super(ThunderNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
