import argparse
import os
import pprint
import shutil
import sys

code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, "lib"))

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

# import _init_paths
import lib.models as models
import lib.datasets as datasets
# from config import config
# from config import update_config
from lib.core.criterion import CrossEntropy
from lib.core.function import train_ee, validate_ee
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, FullEEModel, get_rank, json_save

import pdb
import time
import subprocess

# NOTE
from tools import io_file
from lib.config.default import _C as cfg_def
from lib.models import model_anytime
from lib.datasets.gta5 import GTA5


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local-rank", type=int, default=0)
    # parser.add_argument('opts',
    #                     help="Modify config options using the command-line",
    #                     default=None,
    #                     nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)
    # NOTE
    cfg_dir, cfg_file = os.path.split(args.cfg)
    cfg_model = io_file.load_yaml(cfg_file, cfg_dir, to_yacs=True)
    cfg_def.merge_from_other_cfg(cfg_model)

    return args, cfg_def


def main():
    args, config = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    if args.local_rank == 0:
        logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    # NOTE
    # model = eval('models.'+config.MODEL.NAME +
    #              '.get_seg_model')(config)
    model = model_anytime.get_seg_model(config)
    Path(os.path.join(final_output_dir, "checkpoints")).mkdir(exist_ok=True, parents=True)

    if args.local_rank == 0:
        with open(f"{final_output_dir}/config.yaml", "w") as f: 
            f.write(config.dump())

        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'code')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib'), os.path.join(models_dst_dir, 'lib'))
        shutil.copytree(os.path.join(this_dir, '../tools'), os.path.join(models_dst_dir, 'tools'))
        # shutil.copytree(os.path.join(this_dir, '.   ./scripts'), os.path.join(models_dst_dir, 'scripts'))
        shutil.copytree(os.path.join(this_dir, '../experiments'), os.path.join(models_dst_dir, 'experiments'))

    if True:
        model.eval()
        dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        dump_output = model.to(device)(dump_input.to(device))

        dump_output = model.to(device)(dump_input.to(device))

        stats = {}
        saved_stats = {}
        for i in range(4):
            setattr(model, f"stop{i+1}", "anY_RanDOM_ThiNg")
            summary, stats[i+1] = get_model_summary(model.to(device), dump_input.to(device), verbose=False)
            delattr(model, f"stop{i+1}")

            if args.local_rank == 0:
                logger.info(f'\n\n>>>>>>>>>>>>>>>>>>>>>>>  EXIT {i+1}  >>>>>>>>>>>>>>>>>>>>>>>>>>  ')
                logger.info(summary)

        saved_stats['params'] = [stats[i+1]['params'] for i in range(4)]
        saved_stats['flops'] = [stats[i+1]['flops'] for i in range(4)]
        saved_stats['counts'] = [stats[i+1]['counts'] for i in range(4)]
        saved_stats['Gflops'] = [f/(1024**3) for f in saved_stats['flops']]
        saved_stats['Mparams'] = [f/(10**6) for f in saved_stats['params']]

        json_save(os.path.join(final_output_dir, 'stats.json'), saved_stats)

    # NOTE
    if True: # distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    #NOTE 
    train_dataset = GTA5(device=device,
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        batch_idx=None, # NOTE
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)

        if distributed:
            extra_train_sampler = DistributedSampler(extra_train_dataset)
        else:
            extra_train_sampler = None

        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # NOTE
    test_dataset = GTA5(device=device,
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        # num_samples=config.TEST.NUM_SAMPLES,
                        batch_idx=(0, 10), # NOTE
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        center_crop_test=config.TEST.CENTER_CROP_TEST,
                        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    criterion = CrossEntropy(
        ignore_label=config.TRAIN.IGNORE_LABEL,
        weight=train_dataset.class_weights
    )

    model = FullEEModel(model, criterion, config=config)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank)

    if config.TRAIN.OPTIMIZER == 'sgd':
        if config.TRAIN.ALLE_ONLY:
            param = [
                {'params': model.module.model.exit1.parameters(), 'lr': config.TRAIN.EXTRA_LR},
                {'params': model.module.model.exit2.parameters(), 'lr': config.TRAIN.EXTRA_LR},
                {'params': model.module.model.exit3.parameters(), 'lr': config.TRAIN.EXTRA_LR},
                {'params': model.module.model.last_layer.parameters(),  'lr': config.TRAIN.EXTRA_LR},
            ]
        elif config.TRAIN.EE_ONLY:
            param = [
                {'params': model.module.model.exit1.parameters(), 'lr': config.TRAIN.EXTRA_LR},
                {'params': model.module.model.exit2.parameters(), 'lr': config.TRAIN.EXTRA_LR},
                {'params': model.module.model.exit3.parameters(), 'lr': config.TRAIN.EXTRA_LR}
                ]
        else:
            param = [
                {'params':
                  filter(lambda p: p.requires_grad,
                         model.parameters()),
                'lr': config.TRAIN.LR}
                  ]

        optimizer = torch.optim.SGD(param,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
            )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        if config.DATASET.EXTRA_TRAIN_SET:
            model_state_file = os.path.join(config.RESUME_DIR, 'checkpoint.pth.tar')
            assert os.path.isfile(model_state_file)
            load_optimizer_dict = False
        else:
            model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')


        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, 
                        map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            if not config.DATASET.EXTRA_TRAIN_SET:
                optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    # NOTE load ADP weights
    if True:
        pretrained_dict = torch.load(config.TEST.MODEL_FILE)
        model_dict = model.module.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict)
        logger.info(f"=> loaded weights from ADP: {config.TEST.MODEL_FILE}")

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * epoch_iters
    
    logger.info('Starting training at rank {}'.format(args.local_rank))
    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= config.TRAIN.END_EPOCH:
            train_ee(config, epoch-config.TRAIN.END_EPOCH, 
                  config.TRAIN.EXTRA_EPOCH, epoch_iters, 
                  config.TRAIN.EXTRA_LR, extra_iters, 
                  extra_trainloader, optimizer, model, 
                  writer_dict, device)
        else:
            train_ee(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict,
                  device)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

            torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'checkpoint.pth'))
            
            # NOTE: save intermediate checkpts
            if epoch % 10 == 0:
                torch.save(model.module.state_dict(), os.path.join(final_output_dir, "checkpoints", f'checkpoint_ep{epoch}.pth'))                        

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                       os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: {}'.format((end-start)/3600))
                logger.info('++++++++++++++ Done ++++++++++++++++')

                # NOTE
                pid = os.getpid()
                torch.cuda.empty_cache()
                # devices = os.environ['CUDA_VISIBLE_DEVICES']
                # device = devices.split(',')[1]
                command = f'CUDA_VISIBLE_DEVICES={device} python LTT-EE-CV/tools/test_ee.py --cfg {final_output_dir}/config.yaml'
                print("For testing run:", command)

                # NOTE: runs validation
                # subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()

"""
python -m torch.distributed.launch LTT-EE-CV/tools/train_ee.py --cfg "LTT-EE-CV/experiments/config/gta5/w18_adpc_train" --local-rank=0
"""
    
