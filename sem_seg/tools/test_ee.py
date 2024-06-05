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

# import _init_paths
import lib.models as models
import lib.datasets as datasets
# from config import config
# from config import update_config
from core.function import testval_ee, testval_ee_profiling, testval_ee_profiling_actual
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, FullEEModel, json_save

import pdb

# NOTE
from tools import io_file
from lib.models import model_anytime
from lib.datasets.gta5 import GTA5


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # parser.add_argument('opts',
    #                     help="Modify config options using the command-line",
    #                     default=None,
    #                     nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # NOTE
    # update_config(config, args)
    cfg_dir, cfg_file = os.path.split(args.cfg)
    cfg = io_file.load_yaml(cfg_file.split(".")[0], cfg_dir, to_yacs=True)

    return args, cfg


def main():

    args, config = parse_args()

    # config.defrost()
    config.OUTPUT_DIR = args.cfg[:-len('config.yaml')]
    
    try:
        if config.TEST.SUB_DIR:
            config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, config.TEST.SUB_DIR)
    except:
        pass
    config.freeze()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')
    
    print(final_output_dir)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # NOTE
    # model = eval('models.'+config.MODEL.NAME +
    #              '.get_seg_model')(config)
    model = model_anytime.get_seg_model(config)

    # device = 0
    device = torch.device('cuda:0')
    model.eval()

    dump_input = torch.rand(
    (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )

    if config.PYRAMID_TEST.USE:
        dump_input = torch.rand(
            (1, 3, config.PYRAMID_TEST.SIZE, config.PYRAMID_TEST.SIZE // 2)
            )
    dump_output = model.to(device)(dump_input.to(device))
    del dump_output
    dump_output = model.to(device)(dump_input.to(device))

    if not (config.MASK.USE and (config.MASK.CRIT == 'conf_thre' or config.MASK.CRIT == 'entropy_thre')):
        stats = {}
        saved_stats = {}
        for i in range(4):
            setattr(model, f"stop{i+1}", "anY_RanDOM_ThiNg")
            summary, stats[i+1] = get_model_summary(model.to(device), dump_input.to(device), verbose=True)
            delattr(model, f"stop{i+1}")

            logger.info(f'\n\n>>>>>>>>>>>>>>>>>>>>>>>  EXIT {i+1}  >>>>>>>>>>>>>>>>>>>>>>>>>>  ')
            logger.info(summary)

        saved_stats['params'] = [stats[i+1]['params'] for i in range(4)]
        saved_stats['flops'] = [stats[i+1]['flops'] for i in range(4)]
        saved_stats['counts'] = [stats[i+1]['counts'] for i in range(4)]
        saved_stats['Gflops'] = [f/(1024**3) for f in saved_stats['flops']]
        saved_stats['Gflops_mean'] = np.mean(saved_stats['Gflops'])
        saved_stats['Mparams'] = [f/(10**6) for f in saved_stats['params']]
        json_save(os.path.join(final_output_dir, 'test_stats.json'), saved_stats)

    # NOTE
    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     model_state_file = os.path.join(final_output_dir,
    #                                     'final_state.pth')
    model_state_file = os.path.join(final_output_dir, 'final_state.pth')

    try:
        if config.TEST.SUB_DIR:
            model_state_file = args.cfg[:-len('config.yaml')] + 'final_state.pth'
    except:
        pass

    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = [0]

    model = nn.DataParallel(model, device_ids=gpus).cuda()
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # NOTE
    test_dataset = GTA5(device=device,
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        # num_samples=config.TEST.NUM_SAMPLES,
                        batch_idx=(0, 200), # NOTE
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()

    if 'val' in config.DATASET.TEST_SET:
        results, _ = testval_ee(config, 
                           test_dataset, 
                           testloader, 
                           model, sv_dir=final_output_dir, sv_pred=True)

        if config.MASK.USE and config.MASK.CRIT == 'conf_thre':
            results_profiling = testval_ee_profiling(config, 
                               test_dataset, 
                               testloader, 
                               model, sv_dir=final_output_dir, sv_pred=True)
            json_save(os.path.join(final_output_dir, 'test_stats.json'), results_profiling)

    mean_IoUs = []
    for i, result in enumerate(results):
        mean_IoU, IoU_array, pixel_acc, mean_acc = result

        msg = 'Exit: {}, MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(i+1, mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)

        mean_IoUs.append(mean_IoU)

    mean_IoUs.append(np.mean(mean_IoUs))
    print_result = '\t'.join(['{:.2f}'.format(m*100) for m in mean_IoUs])
    result_file_name = f'{final_output_dir}/result.txt'

    with open(result_file_name, 'w') as f:
        f.write(print_result) 

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end - start) / 60))
    logger.info('Done')
    logging.info(print_result)


if __name__ == '__main__':
    main()