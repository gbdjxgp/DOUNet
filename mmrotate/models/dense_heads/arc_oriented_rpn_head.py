# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.core import anchor_inside_flags, unmap

from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .oriented_rpn_head import OrientedRPNHead
from ..backbones.modules import AdaptiveRotatedConv2d, RountingFunction,rotate_conv_kernel


@ROTATED_HEADS.register_module()
class ARC_OrientedRPNHead(OrientedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = AdaptiveRotatedConv2d(
            self.in_channels, self.feat_channels, 3, padding=1,rounting_func=RountingFunction(
                in_channels=self.in_channels,
                kernel_number=4,
            ),kernel_number=4,rotate_func=rotate_conv_kernel
        )
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        # self.rpn_cls = AdaptiveRotatedConv2d(
        #     self.feat_channels, self.num_anchors * self.cls_out_channels, 1, padding=1,rounting_func=RountingFunction(
        #         in_channels=self.in_channels,
        #         kernel_number=4,
        #     ),kernel_number=4,rotate_func=rotate_conv_kernel
        # )
        # self.rpn_reg = AdaptiveRotatedConv2d(
        #     self.feat_channels, self.num_anchors * 6, 1, padding=1,rounting_func=RountingFunction(
        #         in_channels=self.in_channels,
        #         kernel_number=4,
        #     ),kernel_number=4,rotate_func=rotate_conv_kernel
        # )
