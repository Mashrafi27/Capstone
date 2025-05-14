import torch

# class CrossStitchUnit(torch.nn.Module):
#     def __init__(self, num_channels):
#         super(CrossStitchUnit, self).__init__()
        
#         # Learnable 2x2 matrix per channel
#         self.alpha = torch.nn.Parameter(torch.eye(2).unsqueeze(-1).unsqueeze(-1))  # Shape (2, 2, 1, 1)

#     def forward(self, feat_yolo, feat_clrnet):
#         """
#         feat_yolo: (B, C, H, W)
#         feat_clrnet: (B, C, H, W)
#         """
#         # Stack along a new dim 0
#         stacked = torch.stack([feat_yolo, feat_clrnet], dim=0)  # (2, B, C, H, W)
        
#         # Broadcast alpha to (2,2,1,1) and multiply
#         mixed = (self.alpha[:, :, None, None] * stacked.unsqueeze(1)).sum(dim=0)  # (2, B, C, H, W)

#         feat_yolo_new = mixed[0]
#         feat_clrnet_new = mixed[1]
        
#         return feat_yolo_new, feat_clrnet_new



class YOLOL(torch.nn.Module):
    def __init__(self, backbone, neck, yolo_head, clrhead):
        super(YOLOL, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.clrhead = clrhead
        
        self.offset_2 = torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.offset_1 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
        # self.offset_0 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        
        
        # self.cross_stitch_0 = CrossStitchUnit(num_channels=128)  # out0
        # self.cross_stitch_1 = CrossStitchUnit(num_channels=256)  # out1
        # self.cross_stitch_2 = CrossStitchUnit(num_channels=512)  # out2
        
    def forward(self, x, task_id):
        # Get the outputs from the neck
        out = self.backbone(x['img'] if isinstance(x, dict) else x)
        out0, out1, out2 = self.neck(out)
        
        # out0_yolo, out0_clr = out0.clone(), out0.clone()
        # out1_yolo, out1_clr = out1.clone(), out1.clone()
        # out2_yolo, out2_clr = out2.clone(), out2.clone()
        
        # out0_yolo, out0_clr = self.cross_stitch_0(out0_yolo, out0_clr)
        # out1_yolo, out1_clr = self.cross_stitch_1(out1_yolo, out1_clr)
        # out2_yolo, out2_clr = self.cross_stitch_2(out2_yolo, out2_clr)
        
        
        # Apply Conv2d layers to each output
        if task_id == 0: # lane detection
            # out0_clr = self.offset_0(out0)
            out1_clr = self.offset_1(out1)
            out2_clr = self.offset_2(out2)
            if self.training:
                output = self.clrhead([out0, out1_clr, out2_clr], batch = x)
            else:
                output = self.clrhead([out0, out1_clr, out2_clr])
        else: # object detection
            output = self.yolo_head([out0, out1, out2])
        
        return output
        
       