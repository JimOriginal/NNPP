from __future__ import print_function
import os
import sys
sys.path.append(r'/mnt/hdd1/jim_main_folder/projs')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts

import numpy as np
from util.util import cal_loss, load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port, AverageMeter, intersectionAndUnionGPU
import time
import logging
import random
from tensorboardX import SummaryWriter
#___________model__________________
#from model.swin_unet.vision_transformer import SwinUnet
from model.swin_unet.swin_unet import SwinUnet
from model.pp_lite_seg.ppliteseg_v3 import PPLiteSegT,PPLiteSegB_H
from model.pp_lite_seg.ppliteseg_v3 import HpmStruct
from model.pidnet.pidnet import get_pred_model
from dataset.map_dataset import MapDataset
from dataset.utils import custom_collate_fn,custom_collate_fn_extended,custom_collate_fn_astar_bold_path
import torch.nn.modules.loss as lossfunc
import cv2

import sys
import dataset.map_sample_withAstar as map_sample
sys.modules['map_sample_withAstar'] = map_sample
from dataset.map_sample_withAstar import MapSample

import random

# -x 230613
from timm.scheduler import CosineLRScheduler
from lr_scheduler.scheduler import CosineAnnealingWarmRestartsDecay
import math
from collections import OrderedDict
TRAIN = '/mnt/hdd1/jim_main_folder/projs/nnpp_swin_mae/src/dataset_code/map_set_1w_BOLD/train'
VALIDATION = '/mnt/hdd1/jim_main_folder/projs/nnpp_swin_mae/src/dataset_code/map_set_1w_BOLD/val'

RESULTS_PATH = '/mnt/hdd1/jim_main_folder/projs/NNPP_restart/model_visual_dir/NNPP_sabold_3pix_pplite_orin'

CONFIG_YAML = '/mnt/hdd1/jim_main_folder/projs/NNPP_restart/config/NNPP_sabold_3pix_pplite_orin_gau65.yaml'

class GaussianRelativePE(nn.Module):
    def __init__(self, side, sigma=None):
        super().__init__()
        if sigma is None:
            sigma = (side / 5)
        self.sigma_square = sigma ** 2
        self.alpha = 1 / (2 * math.pi * self.sigma_square)
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer('coord_r', coord_r)
        self.register_buffer('coord_c', coord_c)

    def forward(self, x, center):
        #print("ff",x.device,center.device)
        pe = self.alpha * torch.exp(- ((self.coord_r.view(1, self.side, self.side) - center[:, 0:1].unsqueeze(1)) ** 2 + \
            (self.coord_c.view(1, self.side, self.side) - center[:, 1:2].unsqueeze(1)) ** 2) / (2 * self.sigma_square))
        pe /= pe.amax(dim=(-1, -2)).view(-1, 1, 1)
        return x + pe.unsqueeze(1)

class EuclideanRelativePE(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.side = side
        coord_r = torch.stack([torch.arange(side) for _ in range(side)])
        coord_c = coord_r.T
        self.register_buffer('coord_r', coord_r)
        self.register_buffer('coord_c', coord_c)

    def forward(self, x, center):
        pe = torch.sqrt((self.coord_r.view(1, self.side, self.side) - center[:, 0:1].unsqueeze(1)) ** 2 + \
            (self.coord_c.view(1, self.side, self.side) - center[:, 1:2].unsqueeze(1)) ** 2)
        pe /= pe.amax(dim=(-1, -2)).view(-1, 1, 1)
        return x + pe.unsqueeze(1)

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    file_handler = logging.FileHandler(os.path.join('checkpoints', args.exp_name, 'main-' + str(int(time.time())) + '.log'))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger


def get_parser():
    parser = argparse.ArgumentParser(description='jim path planning')
    parser.add_argument('--config', type=str, default=CONFIG_YAML, help='config file')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['classes'] = cfg.get('classes', 1)
    cfg['sync_bn'] = cfg.get('sync_bn', True)
    cfg['dist_url'] = cfg.get('dist_url', 'tcp://127.0.0.1:6789')
    cfg['dist_backend'] = cfg.get('dist_backend', 'nccl')
    cfg['multiprocessing_distributed'] = cfg.get('multiprocessing_distributed', True)
    cfg['world_size'] = cfg.get('world_size', 1)
    cfg['rank'] = cfg.get('rank', 0)
    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['workers'] = cfg.get('workers', 64)
    cfg['print_freq'] = cfg.get('print_freq', 10)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


# weight initialization:
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rate_pidnet(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def train(gpu, ngpus_per_node):

    # ============= Model ===================
    #if args.arch == 'pplite':
    params = {"in_channels": 3, "num_classes": 1}
    arch_params = HpmStruct(**params)
    model = PPLiteSegB_H(arch_params)
    #model = SwinUnet(img_size=256,num_classes=1)
    #model = get_pred_model(name='pidnet_s', num_classes=1)
    model.apply(weight_init)

    if main_process():
        logger.info(model)

    if args.sync_bn and args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=False)
    else:
        model = torch.nn.DataParallel(model.cuda())

    # =========== Dataloader =================
    train_data = MapDataset(TRAIN)
    test_data = MapDataset(VALIDATION)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers,collate_fn=custom_collate_fn_astar_bold_path, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers,collate_fn=custom_collate_fn_extended,pin_memory=True,sampler=test_sampler)

    # ============= Optimizer ===================
    if main_process():
        logger.info("Use AdamW")
        logger.info("model parameters: {}".format(count_parameters(model)))
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = CosineAnnealingWarmRestartsDecay(opt, T_0=10, T_mult=2, eta_min=1e-6)
    # scheduler = CosineLRScheduler(opt,
    #                               t_initial=args.epochs,  # 300
    #                               cycle_mul=1,
    #                               lr_min=1e-6,
    #                               cycle_decay=0.1,
    #                               warmup_lr_init=1e-6,
    #                               warmup_t=args.warmup_epochs,  # 10
    #                               cycle_limit=1,
    #                               t_in_epochs=True)
 
    #criterion=CustomBCELoss()
    criterion = nn.BCELoss()
    criterion_pass = torch.nn.L1Loss()
    best_test_loss = 1000.0
    start_epoch = 0
    #loss_scaler = NativeScaler()
    #epoch_iters_pidnet = int(train_data.__len__() / args.batch_size /8 / 8)
    # ============= Training from scratch=================
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(train_loader, model, opt, epoch,scheduler,criterion,args=args)

        test_loss = test_epoch(test_loader, model, epoch, criterion)

        if test_loss <= best_test_loss and main_process():
            best_test_loss = test_loss
            logger.info('MIN Loss:%.6f' % best_test_loss)
            torch.save(model.state_dict(), 'checkpoints/%s/best_model.t7' % args.exp_name)  # save the best model
    if main_process():
        logger.info('MIN Loss:%.6f' % best_test_loss)


def train_epoch(train_loader, model, opt, epoch,scheduler,criterion,args):
    train_loss = 0.0
    bce_loss = 0.0
    count = 0.0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_bce_meter=AverageMeter()
    loss_cost_pass_meter=AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    for ii, (map, start, goal,astar_path) in enumerate(train_loader):
        data_time.update(time.time() - end)

        map=map.cuda(non_blocking=True)
        start=start.cuda(non_blocking=True)
        goal=goal.cuda(non_blocking=True)
        #path=path.cuda(non_blocking=True)
        astar_path=astar_path.cuda(non_blocking=True).to(torch.float32)#b,256,256
        batch_size = map.size(0)
        end2 = time.time()
        # map_scale=map/255#map b,1,256,256 map_scale b,1,256,256
        # map_scale=map_scale.squeeze(1)#b,256,256
        # map_scale=map_scale*astar_path+(1-astar_path)*10
        # map_scale=map_scale.unsqueeze(1)#b,1,256,256
        pe = GaussianRelativePE(256,sigma=75).cuda()
        zeros = torch.zeros_like(map, device=map.device, dtype=map.dtype)
        pe_start = pe(zeros, start).cuda(non_blocking=True)
        pe_goal = pe(zeros, goal).cuda(non_blocking=True)
        data_sample = torch.cat([map, pe_start, pe_goal], dim=1).cuda(non_blocking=True)
        logits= model(data_sample)
        label=astar_path
        
        label=label.to(torch.float32).cuda(non_blocking=True)
        loss_BCE = criterion(logits.squeeze(1), label)


        loss_all=loss_BCE
        # loss_all = loss_BCE
        forward_time.update(time.time() - end2)

        preds = logits.max(dim=1)[1]

        if not args.multiprocessing_distributed:
            loss_all = torch.mean(loss_all)

        end3 = time.time()
        opt.zero_grad()
        loss_all.backward()   # the own loss of each process, backward by the optimizer belongs to this process
        opt.step()
        backward_time.update(time.time() - end3)

        # Loss
        if args.multiprocessing_distributed:
            loss=loss_all
            loss = loss * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)  # b_size on one process
            dist.all_reduce(loss), dist.all_reduce(_count)  # obtain the sum of all xxx at all processes
            n = _count.item()
            loss = loss / n   # avg loss across all processes

            loss_bce = loss_BCE * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)  # b_size on one process
            dist.all_reduce(loss_bce), dist.all_reduce(_count)  # obtain the sum of all xxx at all processes
            n = _count.item()
            loss_bce = loss_bce / n   # avg loss across all processes

        # then calculate loss same as without dist
        count += batch_size
        train_loss += loss.item() * batch_size
        bce_loss += loss_BCE.item() * batch_size

        loss_meter.update(loss.item(), batch_size)
        loss_bce_meter.update(loss_bce.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + ii + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (ii + 1) % args.print_freq == 0 and main_process():

            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Forward {for_time.val:.3f} ({for_time.avg:.3f}) '
                        'Backward {back_time.val:.3f} ({back_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss_all {loss_meter.val:.4f} '
                        'Loss_bce {loss_bce.val:.4f} '
                        .format(epoch + 1, args.epochs, ii + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        for_time = forward_time,
                                                        back_time = backward_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        loss_bce=loss_bce_meter
                                                        ))
    scheduler.step(epoch)


    outstr = 'Train %d, loss: %.6f' \
                                     % (epoch + 1,
                                      train_loss * 1.0 / count)

    if main_process():
        logger.info(outstr)
        # Write to tensorboard
        writer.add_scalar('loss_train', train_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('loss_bce_train', bce_loss * 1.0 / count, epoch + 1)

#------------add jim 0606
#|||||||||||||||||||||||||||||||
def refine_100_folder(folder_path):
    max_files = 20
    # Get a list of all the files in the source folder
    files = os.listdir(folder_path)

    # Check if the number of files exceeds the maximum
    if len(files) > max_files:
        # Calculate the number of files to be removed
        num_to_remove = len(files) - max_files
        
        # Randomly select files to remove
        files_to_remove = random.sample(files, num_to_remove)
        
        # Remove each selected file
        for file_name in files_to_remove:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass


def draw_result_pic(model_out,map,start,goal,path,path_to_print,file_name):
    #print("\tdraw model result")
    
    
    sep = np.zeros((map.shape[-1], 128, 3), dtype=np.uint8)
    sep[:, 20:31] = np.array((0, 0, 0))
    map, model_out = map.squeeze(), model_out.squeeze()
    m = map.cpu().detach().numpy()

    #model_solution = MapSample.get_bgr_map(m, start, goal, path_to_print.numpy())
    #print("m shape:",m.shape)
    gt_solution = MapSample.get_orin_map(m, start, goal, torch.nonzero(path > 0).cpu().numpy())
    #md_out=mod_o(model_out)
    md_out=model_out.cpu().detach().numpy()
    model_rainbow=cv2.applyColorMap((md_out*255).astype('uint8'),cv2.COLORMAP_JET)
    comparison = np.concatenate((model_rainbow, sep, gt_solution), axis=1)
    filename = os.path.join(os.path.abspath(RESULTS_PATH), file_name + '.png')
    #filename = os.path.join(RESULTS_PATH, str(int(time())) + '.png')
    cv2.imwrite(filename, comparison)

#finishing add jim 0606
#|||||||||||||||||||||||||||||||
def test_epoch(test_loader, model, epoch, criterion):
    test_loss = 0.0
    count = 0.0
    model.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for map, start, goal,astar_path, filenames,_ in test_loader:
        map=map.cuda(non_blocking=True)
        start=start.cuda(non_blocking=True)
        goal=goal.cuda(non_blocking=True)
        #path=path.cuda(non_blocking=True)
        astar_path=astar_path.cuda(non_blocking=True).to(torch.float32)
        if start.dtype != torch.long:
                start, goal = start.long(), goal.long()
        # data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True).squeeze(1)
        batch_size = map.size(0)

        pe = GaussianRelativePE(256,sigma=75).cuda()
        zeros = torch.zeros_like(map, device=map.device, dtype=map.dtype)
        pe_start = pe(zeros, start).cuda(non_blocking=True)
        pe_goal = pe(zeros, goal).cuda(non_blocking=True)
        data_sample = torch.cat([map, pe_start, pe_goal], dim=1).cuda(non_blocking=True)
        logits = model(data_sample)
        # data_cost_pass=path.clone()*map
        # cost_label = torch.sum(data_cost_pass, dim=1)  # b,1
        label = astar_path
        label=label.to(torch.float32).cuda(non_blocking=True)
        loss_BCE = criterion(logits.squeeze(1), label)


        loss_all=loss_BCE
        
        
        if args.multiprocessing_distributed:
            loss_all = loss_all * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)
            dist.all_reduce(loss_all), dist.all_reduce(_count)
            n = _count.item()
            loss_all = loss_all / n

            loss_bce = loss_BCE * batch_size
            _count = label.new_tensor([batch_size], dtype=torch.long).cuda(non_blocking=True)  # b_size on one process
            dist.all_reduce(loss_bce), dist.all_reduce(_count)  # obtain the sum of all xxx at all processes
            n = _count.item()
            loss_bce = loss_bce / n   # avg loss across all processes





        else:
            loss_all = torch.mean(loss_all)
        #print figure
        if epoch%2==0:
            for m, out, s, g, p,fn in zip(map, logits, start, goal, astar_path,filenames):
                file_name=fn.split(os.sep)[-1].split('.')[-2]
                #print("m size:",m.size(),"out size:",out.size(),"s size:",s.size(),"g size:",g.size(),"p size:",p.size(),"fn:",file_name)
                refine_100_folder(RESULTS_PATH)
                draw_result_pic(out,m,s,g,p,p,file_name)
        preds = logits.max(dim=1)[1]

        count += batch_size
        test_loss += loss_all.item() * batch_size

        intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)


    outstr = 'Test %d, loss: %.6f' % (epoch + 1,
                                     test_loss * 1.0 / count,
                                     )

    if main_process():
        logger.info(outstr)
        # Write to tensorboard
        writer.add_scalar('loss_test', test_loss * 1.0 / count, epoch + 1)
        writer.add_scalar('loss_bce_test', loss_bce * 1.0 / count, epoch + 1)

    return test_loss * 1.0 / count



def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    if main_process():
        if not os.path.exists('checkpoints'):
            os.makedirs('./checkpoints')
        if not os.path.exists('checkpoints/' + args.exp_name):
            os.makedirs('./checkpoints/' + args.exp_name)

        global logger, writer
        writer = SummaryWriter('checkpoints/' + args.exp_name)
        logger = get_logger()
        logger.info(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert not args.eval, "The all_reduce function of PyTorch DDP will ignore/repeat inputs " \
                          "(leading to the wrong result), " \
                          "please use main.py to test (avoid DDP) for getting the right result."
    train(gpu, ngpus_per_node)
    if main_process():
        writer.close()
    if args.distributed and main_process():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_parser()
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    #args.gpu = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES=0,1'].split(',')]
    args.gpu = [0,1,2,3,4,5,6,7]
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.gpu)
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, args.ngpus_per_node, args)

