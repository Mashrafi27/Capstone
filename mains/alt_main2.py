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


from multitask import YOLOL


def main():
    
    device = 'cuda:2'
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
    args.batch_size = 32
    args.local_rank = 0
    args.world_size = 1
    args.epoch = 100
    
    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
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
    
    net = YOLOL(darknet_backbone, darknet_neck, darknet_head, clrnet_head)
    
    for param in net.backbone.parameters():
        param.requires_grad = True
    for param in net.neck.parameters():
        param.requires_grad = True
    for param in net.yolo_head.parameters():
        param.requires_grad = True
    
    
    recorder = build_recorder(cfg)
    
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    net = net.to(device)

    # optimizer = build_optimizer(cfg, net)
    optimizer = PCGrad(optim.Adam(net.parameters())) 
    
    
    # weight_opt = FAMO(n_tasks=2, device=device)
    # opt = torch.optim.Adam(net.parameters())
    
    scheduler = build_scheduler(cfg, optimizer.optimizer)
    criterion_obj = util.ComputeLoss(net, params)

    loss_scaler = 0.1
    
    best_obj = 0
    f_loss = open("all_loss_pcgrad_control2_v8n.txt", "a")
    f_scores = open("all_scores_pcgrad_control2_v8n.txt", "a")
    
    
    
    net.eval()
    test_obj(args, params, device, net)
    

    # f_obj = open('./weights/step_obj.csv', 'w')
    # f_lane = open('./weights/step_lane.csv', 'w')
    for epoch in range(0, 15):
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
            x = i + max_iter #* epoch
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
        
        
        
            # total_loss = loss_cls_obj + loss_box_obj + loss_dfl_obj + loss_scaler*loss_lane
            f_loss.write(f"{loss_cls_obj.item():.6f}, {loss_box_obj.item():.6f}, {loss_dfl_obj.item():.6f}, {output_lane['loss_stats']['cls_loss'].item():.6f}, {output_lane['loss_stats']['reg_xytl_loss'].item():.6f}, {output_lane['loss_stats']['seg_loss'].item():.6f}, {output_lane['loss_stats']['iou_loss'].item():.6f}\n")
            p_bar.set_description(f"{loss_cls_obj.item():.6f}, {loss_box_obj.item():.6f}, {loss_dfl_obj.item():.6f}, {output_lane['loss_stats']['cls_loss'].item():.6f}, {output_lane['loss_stats']['reg_xytl_loss'].item():.6f}, {output_lane['loss_stats']['seg_loss'].item():.6f}, {output_lane['loss_stats']['iou_loss'].item():.6f}")
            
            # total_loss = torch.stack([loss_lane, loss_cls_obj + loss_box_obj + loss_dfl_obj])
            total_loss = [loss_lane, loss_cls_obj + loss_box_obj + loss_dfl_obj]
            
            # opt.zero_grad()
            # weight_opt.backward(total_loss)
            # opt.step()
            
            # with torch.no_grad():
            #     weight_opt.update(total_loss)
            optimizer.pc_backward(total_loss) # calculate the gradient can apply gradient modification
            optimizer.step()
            # amp_scale.scale(total_loss).backward()
            
            # if x % accumulate == 0:
            #     amp_scale.unscale_(optimizer)  # unscale gradients
            #     util.clip_gradients(net)  # clip gradients
            #     amp_scale.step(optimizer)  # optimizer.step
            #     amp_scale.update()
            #     optimizer.zero_grad()
            # optimizer.step()
            # del loss_lane, loss_obj, total_loss, output_lane, output_obj, samples, targets, data_lane

        scheduler.step()
        
        net.eval()
        last_obj = test_obj(args, params, device, net)
        # metric_lane = validate_lane(cfg, net, recorder)
        # validate_lane(cfg, net, recorder, device)
        
        p_bar.set_description(f'''mAP: {last_obj[5]:.3f}',
                                 epoch: {(epoch + 1)},
                                 mAP@50: {last_obj[4]:.3f}''')
        
        if last_obj[5] > best_obj:
            best_obj = last_obj[5]
            
        torch.save(net.state_dict(), './weights/pcgrad_control_v8n_model2_last.pt')
        if best_obj == last_obj[5]:
            torch.save(net.state_dict(), './weights/pcgrad_control_v8n_model2_best.pt')
        
        # f_scores.write(f"tp: {last_obj[0]}, fp: {last_obj[1]}, precision: {last_obj[2]}, recall: {last_obj[3]}, mAP@50: {last_obj[4]}, mAP: {last_obj[5]}\n")
        f_scores.write(f"{last_obj[2]}, {last_obj[3]}, {last_obj[4]}, {last_obj[5]}\n")

            
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
    return tp, fp, m_pre, m_rec, map50, mean_ap

@torch.no_grad()
def validate_lane(cfg, net, recorder, device):
    
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
    recorder.logger.info('metric: ' + str(metric))
    # return metric
    
if __name__ == '__main__':
    main()
