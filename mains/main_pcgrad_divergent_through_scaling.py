import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

import yaml
import os
import glob
from torch.utils import data
import torch.optim as optim

from PCGrad.pcgrad import PCGrad
from FAMO.famo import FAMO

from clrnet.models.registry import build_net
from clrnet.engine.registry import build_trainer, build_evaluator
from clrnet.engine.optimizer import build_optimizer
from clrnet.engine.scheduler import build_scheduler
from clrnet.datasets import build_dataloader
from clrnet.utils.recorder import build_recorder
from clrnet.utils.net_utils import save_model, load_network, resume_network
from clrnet.utils.recorder import build_recorder
from mmcv.parallel import MMDataParallel
from nets.nn import DarkNet
from utils import util

from collections import OrderedDict


# yolo

from utils.dataset import Dataset
from clrnet.utils.config import Config
import argparse


from multitask_cross_stitch import YOLOL
# from multitask import YOLOL
import math

import wandb

ablation = "pcgrad_divergent_through_scaling"


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0


# def learning_rate(args, params):
#     def fn(epoch):
#         # Linear warmup for 3 epochs, then cosine decay
#         warmup_epochs = 3
#         if epoch < warmup_epochs:
#             return epoch / warmup_epochs
#         else:
#             return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
#     return fn

def main():
    
    wandb.init(project="cross_stitch_model_scaled_v8n")
    
    device = 'cuda:1'
    cfg = Config.fromfile("configs/clrnet/clr_resnet18_culane.py")
    cfg.load_from = None
    cfg.resume_from = None
    cfg.finetune_from = None
    cfg.view = None
    cfg.seed = 0
    cfg.gpus = 1
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    train_loader_lane = build_dataloader(cfg.dataset.train,
                                        cfg,
                                        is_train=True)
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.input_size = 640
    args.batch_size = 18
    args.local_rank = 0
    args.world_size = 1
    args.epochs = 30
    
    with open(os.path.join('utils', f'args_pcgrad_cross_stitch.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    folder_path = "/data/bdd100k/images/train2017/"
    filenames = glob.glob(os.path.join(folder_path, '*'))

    train_dataset_obj = Dataset(filenames, args.input_size, params, True)
    train_loader_obj = data.DataLoader(train_dataset_obj, 18, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    
    state_dict = torch.load("/home/mmm9886/Desktop/Capstone_Implementation/YOLOv8-pt copy/weights/best.pt", map_location=torch.device(device))
    darknet_backbone = state_dict['model'].net
    darknet_backbone = darknet_backbone.float()
    darknet_backbone = darknet_backbone.to(device)
    darknet_neck = state_dict['model'].fpn
    darknet_neck = darknet_neck.float()
    darknet_neck = darknet_neck.to(device)
    darknet_head = state_dict['model'].head
    darknet_head = darknet_head.float()
    darknet_head = darknet_head.to(device)
    
    net = build_net(cfg)
    #clrnet = torch.load('/home/mmm9886/Desktop/Capstone_Implementation/clrnet/work_dirs/clr/r18_culane/20250111_141413_lr_6e-04_b_24/ckpt/9.pth', map_location=torch.device("cuda:1"))
    #prefix_to_keep = "module.heads."
    #filtered_od = OrderedDict((k, v) for k, v in clrnet["net"].items() if k.startswith(prefix_to_keep))
    #filtered_od = {k.replace("module.heads.", ""): v for k, v in filtered_od.items()}
    clrnet_head = net.heads
    
    clrnet_head = clrnet_head.to(device)
    #clrnet_head.load_state_dict(filtered_od)
    
    recorder = build_recorder(cfg)
    
    net = YOLOL(darknet_backbone, darknet_head, clrnet_head)
    # net = YOLOL(darknet_backbone, darknet_neck, darknet_head, clrnet_head)
    
    # net.load_state_dict(torch.load('/home/mmm9886/Desktop/Capstone_Implementation/clrnet/weights/pcgrad_divergentneck_cross_stitch_model_scaled_aggressive_epoch2_v8n.pt'))
    

    
    # net.eval()
    # net = net.to(device)
    # last_obj = test_obj(args, params, device, net)
    # metric_lane = validate_lane(cfg, net, recorder, 'cuda:0')
    net = net.to(device)
    
    # for param in net.backbone.parameters():
    #     param.requires_grad = True
    # for param in net.neck.parameters():
    #     param.requires_grad = True
    # for param in net.yolo_head.parameters():
    #     param.requires_grad = True
    
    
    
    
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64
    # net = net.to(device)
    
        
    for param in net.parameters():
        param.requires_grad = True
        
    # # net.apply(reset_weights)
    
    # p = [], [], []
    # for v in net.modules():
    #     if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
    #         p[2].append(v.bias)
    #     if isinstance(v, torch.nn.BatchNorm2d):
    #         p[1].append(v.weight)
    #     elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
    #         p[0].append(v.weight)
    # # p[0].append(net.neck.cross_stitch_h4.alpha)


    # # optimizer = build_optimizer(cfg, net)
    # optimizer = PCGrad(optim.Adam(net.parameters(), lr = 0.0001, betas = (0.9, 0.937), weight_decay = 0.0005))
    # optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    # optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    # optimizer.add_param_group({'params': p[1]})
    
    # optimizer_clr = optim.Adam(net.clrhead.parameters(), lr = 0.01)
    # optimizer_clr.add_param_group({'params': net.offset_1.parameters()})
    # optimizer_clr.add_param_group({'params': net.offset_2.parameters()})
    
    optimizer = PCGrad(optim.Adam(net.parameters(), lr = 0.001))
    
    
    # used_params = set(map(id, p[0] + p[1] + p[2]))
    all_params = list(net.named_parameters())

    # print("Missing parameters:")
    # for name, param in all_params:
    #     if id(param) not in used_params:
    #         print(f"- {name}, shape: {tuple(param.shape)}, requires_grad: {param.requires_grad}")
    
    # del p
    # optimizer = PCGrad(optimizer)
    # for group in optimizer.param_groups:
    #     group["lr"] = params["lr0"]               # Set base LR
    #     group["initial_lr"] = params["lr0"]       # Required for LambdaLR to scale correctly
        
    # for i, group in enumerate(optimizer.optimizer.param_groups):
    #     print(f"group {i} lr: {group['lr']}, initial_lr: {group.get('initial_lr')}")
    
    lr = learning_rate(args, params)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lr)
    # scheduler_clr = build_scheduler(cfg, optimizer)
    criterion_obj = util.ComputeLoss(net, params)
    
    # for i, group in enumerate(optimizer.optimizer.param_groups):
    #     print(f"after scheduler; group {i} lr: {group['lr']}, initial_lr: {group.get('initial_lr')}")

    loss_scaler = 0.1
    
    best_obj = 0
    f_loss = open(f"losses/ablation_{ablation}.txt", "a")
    f_scores = open(f"scores/ablation_{ablation}.txt", "a")
    
    
    step = 0
    
    metric_lane = {'normal': {0.5: {'TP': 1468,
   'FP': 33214,
   'FN': 31309,
   'Precision': 0.042327432097341564,
   'Recall': 0.04478750343228483,
   'F1': 0.89345}},
 'crowd': {0.5: {'TP': 1090,
   'FP': 27733,
   'FN': 26913,
   'Precision': 0.03781702112895951,
   'Recall': 0.03892440095704032,
   'F1': 0.712}},
 'hlight': {0.5: {'TP': 71,
   'FP': 1511,
   'FN': 1614,
   'Precision': 0.04487989886219975,
   'Recall': 0.042136498516320474,
   'F1': 0.623}},
 'shadow': {0.5: {'TP': 93,
   'FP': 2946,
   'FN': 2783,
   'Precision': 0.03060217176702863,
   'Recall': 0.032336578581363004,
   'F1': 0.722}},
 'noline': {0.5: {'TP': 611,
   'FP': 12661,
   'FN': 13410,
   'Precision': 0.046036769138034964,
   'Recall': 0.043577490906497394,
   'F1': 0.46}},
 'arrow': {0.5: {'TP': 153,
   'FP': 2979,
   'FN': 3029,
   'Precision': 0.04885057471264368,
   'Recall': 0.04808296668761785,
   'F1': 0.834}},
 'curve': {0.5: {'TP': 31,
   'FP': 1122,
   'FN': 1281,
   'Precision': 0.026886383347788378,
   'Recall': 0.023628048780487805,
   'F1': 0.687}},
 'cross': {0.5: {'TP': 0,
   'FP': 3478,
   'FN': 0,
   'Precision': 0,
   'Recall': 0,
   'F1': 0}},
 'night': {0.5: {'TP': 870,
   'FP': 21137,
   'FN': 20160,
   'Precision': 0.03953287590312173,
   'Recall': 0.04136947218259629,
   'F1': 0.673}}}
    
    # wandb.log({
    #     "metrics/precision": 0.641, #last_obj[0],
    #     "metrics/recall": 0.332, #last_obj[1],
    #     "metrics/mAP@50": 0.502, #last_obj[2],
    #     "metrics/mAP@50:95": 0.31, #last_obj[3]
    # }, step=step)
    
    wandb.log({
        "metrics/precision": 0, #last_obj[0],
        "metrics/recall": 0, #last_obj[1],
        "metrics/mAP@50": 0, #last_obj[2],
        "metrics/mAP@50:95": 0#last_obj[3]
    }, step=step)
    
    for category, vals in metric_lane.items():
        f1 = vals[0.5].get('F1', None)
        if f1 is not None:
            f1 = 0
            wandb.log({f"lane_metrics/{category}/F1": f1}, step = step)
    

    

    # f_obj = open('./weights/step_obj.csv', 'w')
    # f_lane = open('./weights/step_lane.csv', 'w')
    for epoch in range(0, args.epochs):
        
        epoch_loss = {
            "det_cls": 0.0,
            "det_box": 0.0,
            "det_dfl": 0.0,
            "lane_cls": 0.0,
            "lane_reg": 0.0,
            "lane_seg": 0.0,  # formerly seg
            "lane_iou": 0.0,  # formerly iou
            "count": 0
        }
        # if epoch == 0:
        #     for params in net.clrhead.parameters():
        #         params.requires_grad = True
        #     for params in net.offset_1.parameters():
        #         params.requires_grad = True
        #     for params in net.offset_2.parameters():
        #         params.requires_grad = True
        # if epoch == 7:
        #     for params in net.parameters():
        #         params.requires_grad = True

        # if epoch == 0:
        #     net.eval()
        #     validate_lane(cfg, net, recorder, 'cpu')
    
        net.train()
        
        net.backbone = net.backbone.to(device)
        net.neck = net.neck.to(device)
        net.clrhead= net.clrhead.to(device)
        net.yolo_head = net.yolo_head.to(device)
        
        
        end = time.time()
        max_iter = len(train_loader_lane)
        


        zipped_dls = zip(train_loader_obj, train_loader_lane)

        p_bar = enumerate(zipped_dls)
        
        
        p_bar = tqdm(p_bar, total=len(train_loader_lane))
        
        # p_bar = enumerate(train_loader_obj) # for checking whether object detection is working properly
        # p_bar = tqdm(p_bar, total=len(train_loader_obj))
        
        # for i, data in enumerate(train_loader_lane):
        for i, (data_obj, data_lane) in p_bar:
        # for i, data_obj in p_bar:
            step = i + max_iter * epoch
            date_time = time.time() - end
            samples = data_obj[0].to(device).float() / 255
            targets = data_obj[1].to(device)
            # data_obj = to_cuda(data_obj)
            data_lane = to_cuda(data_lane, device)
            output_obj = net(samples, 1)
            
            
            # experiment = net2(data_lane)
            output_lane = net(data_lane, 0)
        
            
            # amp_scale = torch.cuda.amp.GradScaler()
            # optimizer.zero_grad()
            loss_lane = output_lane['loss'].sum()            
            loss_cls_obj, loss_box_obj, loss_dfl_obj = criterion_obj(output_obj, targets)
            
            # loss_cls_obj_meter.update(loss_cls_obj.item())
            # loss_box_obj_meter.update(loss_box_obj.item())
            # loss_dfl_obj_meter.update(loss_dfl_obj.item())
            # loss_lane_cls_meter.update(output_lane['loss_stats']['cls_loss'].item())
            # loss_lane_reg_meter.update(output_lane['loss_stats']['reg_xytl_loss'].item())
            # loss_seg_meter.update(output_lane['loss_stats']['seg_loss'].item())
            # loss_iou_meter.update(output_lane['loss_stats']['iou_loss'].item())
            
            epoch_loss["det_cls"] += loss_cls_obj.item()
            epoch_loss["det_box"] += loss_box_obj.item()
            epoch_loss["det_dfl"] += loss_dfl_obj.item()

            epoch_loss["lane_cls"] += output_lane['loss_stats']['cls_loss'].item()
            epoch_loss["lane_reg"] += output_lane['loss_stats']['reg_xytl_loss'].item()
            epoch_loss["lane_seg"] += output_lane['loss_stats']['seg_loss'].item()
            epoch_loss["lane_iou"] += output_lane['loss_stats']['iou_loss'].item()

            epoch_loss["count"] += 1

        
        
        
            # total_loss = loss_cls_obj + loss_box_obj + loss_dfl_obj + loss_scaler*loss_lane
            f_loss.write(f"{loss_cls_obj.item():.6f}, {loss_box_obj.item():.6f}, {loss_dfl_obj.item():.6f}, {output_lane['loss_stats']['cls_loss'].item():.6f}, {output_lane['loss_stats']['reg_xytl_loss'].item():.6f}, {output_lane['loss_stats']['seg_loss'].item():.6f}, {output_lane['loss_stats']['iou_loss'].item():.6f}\n")
            p_bar.set_description(f"{loss_cls_obj.item():.6f}, {loss_box_obj.item():.6f}, {loss_dfl_obj.item():.6f}, {output_lane['loss_stats']['cls_loss'].item():.6f}, {output_lane['loss_stats']['reg_xytl_loss'].item():.6f}, {output_lane['loss_stats']['seg_loss'].item():.6f}, {output_lane['loss_stats']['iou_loss'].item():.6f}")
            
            loss_lane_scaled = loss_lane * loss_scaler
            total_loss = [loss_lane_scaled, loss_cls_obj+loss_box_obj+loss_dfl_obj]
            # total_loss = loss_cls_obj + loss_box_obj + loss_dfl_obj + loss_lane

# 
            optimizer.pc_backward(total_loss) # calculate the gradient can apply gradient modification
            
            for name, param in net.named_parameters():
                if param.grad is not None:
                    wandb.log({f"grad_norm/{name}": param.grad.data.norm(2).item()}, step=step)
            # loss *= args.batch_size
            # loss *= args.world_size  # optional if you're using DDP
            optimizer.step()  # apply gradient modification
            optimizer.zero_grad()

            # # Backward
            # amp_scale.scale(total_loss).backward()

            # # Optimization step
            
            # if step % accumulate == 0:
            #     # if epoch < 7:
            #     #     amp_scale.unscale_(optimizer_clr)
            #     #     torch.nn.utils.clip_grad_norm_(net.clrhead.parameters(), max_norm=10.0)  # gradient clipping
            #     #     # Log gradients
            #     #     for name, param in net.named_parameters():
            #     #         if param.grad is not None:
            #     #             wandb.log({f"grad_norm/{name}": param.grad.data.norm(2).item()}, step=step)
            #     #     amp_scale.step(optimizer_clr)
            #     #     amp_scale.update()
            #     #     optimizer_clr.zero_grad()
            #     # else:
            #     amp_scale.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # gradient clipping
            #     # Log gradients
            #     for name, param in net.named_parameters():
            #         if param.grad is not None:
            #             wandb.log({f"grad_norm/{name}": param.grad.data.norm(2).item()}, step=step)
            #     amp_scale.step(optimizer)
            #     amp_scale.update()
            #     optimizer.zero_grad()
                
                
            # amp_scale.scale(total_loss).backward()
            
            # if x % accumulate == 0:
            #     amp_scale.unscale_(optimizer)  # unscale gradients
            #     util.clip_gradients(net)  # clip gradients
            #     amp_scale.step(optimizer)  # optimizer.step
            #     amp_scale.update()
            #     optimizer.zero_grad()
            # optimizer.step()
            # del loss_lane, loss_obj, total_loss, output_lane, output_obj, samples, targets, data_lane
            
            
 
                
            # Log losses and metrics
            if step % 10 == 0:  # Reduce frequency
                wandb.log({
                    "batch_loss/det_cls": loss_cls_obj.item(),
                    "batch_loss/det_box": loss_box_obj.item(),
                    "batch_loss/det_dfl": loss_dfl_obj.item(),

                    "batch_loss/lane_cls": output_lane['loss_stats']['cls_loss'].item(),
                    "batch_loss/lane_reg": output_lane['loss_stats']['reg_xytl_loss'].item(),
                    "batch_loss/lane_seg": output_lane['loss_stats']['seg_loss'].item(),
                    "batch_loss/lane_iou": output_lane['loss_stats']['iou_loss'].item(),
                    
                    "batch_loss/lane_total": output_lane['loss'].sum().item(), 
                    "batch_loss/det_total": loss_cls_obj.item() + loss_box_obj.item() + loss_dfl_obj.item(),
                    "batch_loss/total": output_lane['loss'].sum().item() + loss_cls_obj.item() + loss_box_obj.item() + loss_dfl_obj.item()
                }, step=step)
            # if step % 10 == 0:   
            #     if epoch < 7:
            #         wandb.log({
            #             "lr/group_0": optimizer_clr.param_groups[0]['lr'],
            #             "lr/group_1": optimizer_clr.param_groups[1]['lr'],
            #             "lr/group_2": optimizer_clr.param_groups[2]['lr']
            #         }, step=step)
            #     else:
            #         wandb.log({
            #             "lr/group_0": optimizer.param_groups[0]['lr'],
            #             "lr/group_1": optimizer.param_groups[1]['lr'],
            #             "lr/group_2": optimizer.param_groups[2]['lr']
            #         }, step=step)



        # if epoch < 7:
        #     scheduler_clr.step()
        # else:
        scheduler.step()
        
        n = max(1, epoch_loss["count"])
        avg = {k: v / n for k, v in epoch_loss.items() if k != "count"}

        lane_total = avg["lane_cls"] + avg["lane_reg"] + avg["lane_seg"] + avg["lane_iou"]
        det_total = avg["det_cls"] + avg["det_box"] + avg["det_dfl"]
        total_loss = lane_total + det_total
        
        wandb.log({
            "epoch_loss/lane_cls": avg["lane_cls"],
            "epoch_loss/lane_reg": avg["lane_reg"],
            "epoch_loss/lane_seg": avg["lane_seg"],
            "epoch_loss/lane_iou": avg["lane_iou"],

            "epoch_loss/det_cls": avg["det_cls"],
            "epoch_loss/det_box": avg["det_box"],
            "epoch_loss/det_dfl": avg["det_dfl"],

            "epoch_loss/lane_total": lane_total,
            "epoch_loss/det_total": det_total,
            "epoch_loss/total": total_loss
        }, step=step)

        
        net.eval()
        last_obj = test_obj(args, params, device, net)
        metric_lane = validate_lane(cfg, net, recorder, 'cuda:0')
        net = net.to(device)
        # validate_lane(cfg, net, recorder, device)
        
        p_bar.set_description(f'''mAP: {last_obj[3]:.3f}',
                                 epoch: {(epoch + 1)},
                                 mAP@50: {last_obj[2]:.3f}''')
        
        if last_obj[3] > best_obj:
            best_obj = last_obj[3]
            

        torch.save(net.state_dict(), f'./weights/ablation_{ablation}_last.pt')
        if best_obj == last_obj[3]:
            torch.save(net.state_dict(), f'./weights/ablation_{ablation}_best.pt')
        
        # f_scores.write(f"tp: {last_obj[0]}, fp: {last_obj[1]}, precision: {last_obj[2]}, recall: {last_obj[3]}, mAP@50: {last_obj[4]}, mAP: {last_obj[5]}\n")
        f_scores.write(f"{last_obj[0]}, {last_obj[1]}, {last_obj[2]}, {last_obj[3]}\n")
        
        precision = last_obj[0]
        recall = last_obj[1]
        map_50 = last_obj[2]
        map_50_95 = last_obj[3]

        wandb.log({
            "metrics/precision": precision,
            "metrics/recall": recall,
            "metrics/mAP@50": map_50,
            "metrics/mAP@50:95": map_50_95
        }, step=step)
        
        for category, vals in metric_lane.items():
            f1 = vals[0.5].get('F1', None)
            if f1 is not None:
                wandb.log({f"lane_metrics/{category}/F1": f1}, step = step)
        # f_obj.write(str(last_obj))
        # f_lane.write(metric_lane)
        
        # torch.save(net.state_dict(), 'weights/final_combined_model.pth')
    
    torch.cuda.empty_cache()
        
        
    
def to_cuda(batch, device = 'cpu'):
    for k in batch:
        if not isinstance(batch[k], torch.Tensor):
            continue
        batch[k] = batch[k].to(device)
    return batch

@torch.no_grad()
def test_obj(args, params, device, model=None):
    # model = .to(device)
    folder_path = "/data/bdd100k/images/val2017/"
    filenames = glob.glob(os.path.join(folder_path, '*'))

    val_dataset_obj = Dataset(filenames, args.input_size, params, False)

    loader = data.DataLoader(val_dataset_obj, 8, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm(loader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
    for samples, targets, shapes in p_bar:
        samples = samples.to(device)
        targets = targets.to(device)
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = model(samples, 1)
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).to(device)  # to pixels
        outputs = util.non_max_suppression(outputs, 0.25, 0.45) # 0.001 originally
        # Metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]
            correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).to(device)

            if output.shape[0] == 0:
                if labels.shape[0]:
                    metrics.append((correct, *torch.zeros((3, 0)).to(device)))
                continue

            detections = output.clone()
            util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

            # Evaluate
            if labels.shape[0]:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for j in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        matches = matches.cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)

    # Print results
    print('%10.3g' * 3 % (m_pre, m_rec, mean_ap))

    # Return results
    model.float()  # for training
    return m_pre, m_rec, map50, mean_ap

@torch.no_grad()
def validate_lane(cfg, net, recorder, device):
    
    net = net.to(device)
    net.backbone = net.backbone.to(device)
    net.neck = net.neck.to(device)
    net.clrhead= net.clrhead.to(device)
    net.yolo_head = net.yolo_head.to(device)
    
    val_loader_lane = build_dataloader(cfg.dataset.val,
                                            cfg,
                                            is_train=False)
    predictions = []
    for i, data in enumerate(tqdm(val_loader_lane, desc=f'Validate')):
        data = to_cuda(data, device)
        with torch.no_grad():
            output = net(data, 0)
            output = net.clrhead.get_lanes(output)
            predictions.extend(output)
        if cfg.view:
            val_loader_lane.dataset.view(output, data['meta'])

    metric = val_loader_lane.dataset.evaluate(predictions,
                                                cfg.work_dir)
    # recorder.logger.info('metric: ' + str(metric))
    return metric
    
if __name__ == '__main__':
    main()
