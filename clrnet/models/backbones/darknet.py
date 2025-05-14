import torch
from torch import nn
import torch.nn.functional as F

from clrnet.models.registry import BACKBONES

@BACKBONES.register_module
class YOLOBackboneWrapper(nn.Module):
    def __init__(self, yolo_backbone, out_conv=False, cfg=None):
        super(YOLOBackboneWrapper, self).__init__()
        self.backbone = yolo_backbone
        self.cfg = cfg
        self.out = None
        if out_conv:
            self.out = nn.Conv2d(
                in_channels=self.backbone.out_channels,
                out_channels=self.cfg.featuremap_out_channel,
                kernel_size=1
            )

    def forward(self, x):
        x = self.backbone(x)
        if self.out:
            x = self.out(x[-1])  # Apply the out layer to the last feature map
        return x