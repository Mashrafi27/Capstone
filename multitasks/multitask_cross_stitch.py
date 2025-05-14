import torch
from nets.nn import Conv, CSP, DarkNet, Head


class CrossStitchUnit(torch.nn.Module):
    def __init__(self, num_channels):
        super(CrossStitchUnit, self).__init__()
        
        # Learnable 2x2 matrix per channel
        self.alpha = torch.nn.Parameter(torch.eye(2).unsqueeze(-1).unsqueeze(-1))  # Shape (2, 2, 1, 1)

    def forward(self, feat_yolo, feat_clrnet):
        """
        feat_yolo: (B, C, H, W)
        feat_clrnet: (B, C, H, W)
        """
        # Stack along a new dim 0
        stacked = torch.stack([feat_yolo, feat_clrnet], dim=0)  # (2, B, C, H, W)
        
        # Broadcast alpha to (2,2,1,1) and multiply
        mixed = (self.alpha[:, :, None, None] * stacked.unsqueeze(1)).sum(dim=0)  # (2, B, C, H, W)

        feat_yolo_new = mixed[0]
        feat_clrnet_new = mixed[1]
        
        return feat_yolo_new, feat_clrnet_new
    
    
class CrossStitchDarkFPN(torch.nn.Module):
    def __init__(self, width = [3, 16, 32, 64, 128, 256], depth = [1, 2, 2]):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        
        self.h4_yolo = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h4_clr = CSP(width[3] + width[4], width[4], depth[0], False)
        
        self.cross_stitch_h4 = CrossStitchUnit(num_channels=width[4])
        
        self.h5_yolo = Conv(width[4], width[4], 3, 2)
        self.h5_clr = Conv(width[4], width[4], 3, 2)
        
        self.h6_yolo = CSP(width[4] + width[5], width[5], depth[0], False)
        self.h6_clr = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x, task = 0):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        
        if self.training:
            h4_yolo = self.h4_yolo(torch.cat([self.h3(h2), h1], 1))
            h4_clr = self.h4_clr(torch.cat([self.h3(h2), h1], 1))
            
            h4_yolo, h4_clr = self.cross_stitch_h4(h4_yolo, h4_clr)
            
            if task == 0: # lane detection
                h6_clr = self.h6_clr(torch.cat([self.h5_clr(h4_clr), p5], 1))
                return h2, h4_clr, h6_clr
            else: # object detection
                h6_yolo = self.h6_yolo(torch.cat([self.h5_yolo(h4_yolo), p5], 1))
                return h2, h4_yolo, h6_yolo
            
        else:
            h4_yolo = self.h4_yolo(torch.cat([self.h3(h2), h1], 1))
            h4_clr = self.h4_clr(torch.cat([self.h3(h2), h1], 1))
            
            h4_yolo, h4_clr = self.cross_stitch_h4(h4_yolo, h4_clr)
            
            
            h6_yolo = self.h6_yolo(torch.cat([self.h5_yolo(h4_yolo), p5], 1))
            h6_clr = self.h6_clr(torch.cat([self.h5_clr(h4_clr), p5], 1))
            
            return h2, h4_yolo, h4_clr, h6_yolo, h6_clr



class YOLOL(torch.nn.Module):
    def __init__(self, backbone, yolo_head, clrhead):
        super(YOLOL, self).__init__()
        # width = [3, 16, 32, 64, 128, 256] 
        # depth = [1, 2, 2]
        self.backbone = backbone
        self.neck = CrossStitchDarkFPN()
        # img_dummy = torch.zeros(1, 3, 256, 256)
        self.yolo_head = yolo_head
        # self.yolo_head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        # self.stride = self.yolo_head.stride
        # self.yolo_head.initialize_biases()
        
        self.clrhead = clrhead
        
        self.offset_2 = torch.nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.offset_1 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
        # self.offset_0 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
    def forward(self, x, task_id):
        # Get the outputs from the neck
        out = self.backbone(x['img'] if isinstance(x, dict) else x)
        if self.training:
            out0, out1, out2 = self.neck(out, task = task_id)
            
            if task_id == 0: # lane detection
                # out0_clr = self.offset_0(out0)
                out1 = self.offset_1(out1)
                out2 = self.offset_2(out2)
                output = self.clrhead([out0, out1, out2], batch = x)
            else: # object detection
                output = self.yolo_head([out0, out1, out2])
        else:
            out0, out1_yolo, out1_clr, out2_yolo, out2_clr = self.neck(out)
            # Apply Conv2d layers to each output
            if task_id == 0: # lane detection
                # out0_clr = self.offset_0(out0)
                out1_clr = self.offset_1(out1_clr)
                out2_clr = self.offset_2(out2_clr)
                if self.training:
                    output = self.clrhead([out0, out1_clr, out2_clr], batch = x)
                else:
                    output = self.clrhead([out0, out1_clr, out2_clr])
            else: # object detection
                output = self.yolo_head([out0, out1_yolo, out2_yolo])
        
        return output
        
       