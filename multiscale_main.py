# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datafunction import MedData_train_onlycnn,MedData_val_onlycnn

from multiscale_engine import train_one_epoch, evaluate
import timm.optim.optim_factory as optim_factory


def get_args_parser():
    parser = argparse.ArgumentParser('Medical segmentation ', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--crop_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')




    # # Model parameters
    # parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # # * Random Erase params
    # parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
    #                     help='Random erase prob (default: 0.25)')
    # parser.add_argument('--remode', type=str, default='pixel',
    #                     help='Random erase mode (default: "pixel")')
    # parser.add_argument('--recount', type=int, default=1,
    #                     help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', action='store_true', default=False,
    #                     help='Do not random erase first (clean) augmentation split')


    # Dataset parameters

    parser.add_argument('--train_source_dir', default='train1/source', type=str,
                        help='train source dir path')
    parser.add_argument('--train_label_dir', default='train1/label', type=str,
                        help='train label dir path')
    parser.add_argument('--valid_source_dir', default='valid1/source', type=str,
                        help='valid source dir path')
    parser.add_argument('--valid_label_dir', default='valid1/label', type=str,
                        help='valid label dir path')
    parser.add_argument('--output_dir', default='./output_dir_without_attention_model_64f_40w_3s_64h',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_without_attention_model_64f_40w_3s_64h',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    # rvos parameters
    parser.add_argument('--cpu', dest = 'use_gpu', action = 'store_false')
    parser.set_defaults(use_gpu = True)
    # parser.add_argument('-base_model', dest = 'base_model', default = 'resnet34',
    #                     choices = ['resnet101', 'resnet50', 'resnet34', 'vgg16'])
    # parser.add_argument('-skip_mode', dest = 'skip_mode', default = 'concat',
    #                     choices = ['sum', 'concat', 'mul', 'none'])
    # parser.add_argument('-hidden_size', dest = 'hidden_size', default = 128, type = int)

    parser.add_argument('--init_feature', dest = 'init_feature', default = 64, type = int)
    parser.add_argument('--scale', dest = 'scale', default = 3, type = int)
    parser.add_argument('--baseWidth', dest = 'baseWidth', default = 40, type = int)
    parser.add_argument('--block_num', dest = 'block_num', default = 2, type = int)
    parser.add_argument('--down_num', dest = 'down_num', default = 3, type = int)
    parser.add_argument('--kernel_size', dest = 'kernel_size', default = 3, type = int)
    parser.add_argument('--dropout', dest = 'dropout', default = 0.0, type = float)
    parser.add_argument('--hidden_size', dest = 'hidden_size', default = 64, type = int)
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = MedData_train_onlycnn(args.train_source_dir,args.train_label_dir,args.crop_size)
    dataset_val = MedData_val_onlycnn(args.valid_source_dir,args.valid_label_dir)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # from model.twoD.OnlyHRNet import get_seg_model
    # from model.twoD.config import HRNet8
    # model = get_seg_model(HRNet8, in_feat=HRNet8.DATASET.NUM_CLASSES).to(device)

    # from model.twoD.unet.unet_model import get_seg_model
    # model = get_seg_model(n_channels=1 , n_classes=1 ).to(device)

    # from model.twoD_rnn.cnnrnnNet import cnnrnnNet
    # model = cnnrnnNet(n_channels = 1,n_classes = 16,bilinear = False, num_blocks = 40).to(device)

    #from model.threeD.unet3d import get_seg_model
    #model= get_seg_model(in_channels = 1, out_channels = 1, init_features = 32).to(device)

    # from model.rvos_new.model import Rvosnet
    # model = Rvosnet(args).to(device)

    from model_1.without_attention.model import Rvosnet
    model = Rvosnet(args).to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Continue train checkpoint from: %s" % args.resume)
        # load resume model
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)



    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()


    from util.loss_function import SoftDiceLoss, BCELoss2d,DiceCeloss
    # criterion = DiceCeloss()
    criterion = SoftDiceLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # if args.eval:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_dice = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.crop_size,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, device,args.crop_size)
        print(f"Dice of the network on the {len(dataset_val)} test images: {test_stats['dice']:.3f}%")
        max_dice = max(max_dice, test_stats["dice"])
        print(f'Max dice: {max_dice:.3f}%')


        if log_writer is not None:
            log_writer.add_scalar('perf/test_dice', test_stats['dice'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
