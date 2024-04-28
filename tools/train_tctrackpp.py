# Copyright (c) SenseTime. All Rights Reserved.

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.static import InputSpec
from paddle.optimizer import SGD
from paddle.optimizer.lr import PiecewiseDecay
from paddle.regularizer import L2Decay
from paddle.metric import Accuracy
from paddle.metric import Loss
from paddle.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, reduce_gradients, average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.datasets.dataset import TrkDataset
from pysot.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='TCTrackpp tracking')
parser.add_argument('--cfg', type=str, default='../experiments/TCTrack++/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

def seed_paddle(seed=0):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    paddle.set_device('gpu')

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS, 
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True,
                              sampler=train_sampler)
    return train_loader

def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.trainable = True
            for m in getattr(model.backbone, layer).sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.trainable = False
        for m in model.backbone.sublayers():
            if isinstance(m, nn.BatchNorm2D):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.trainable,
                                            model.backbone.parameters()),
                          'learning_rate': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    # trainable_params += [{'params': model.att.parameters(),
    #                       'learning_rate': cfg.TRAIN.BASE_LR}]
    
    # trainable_params += [{'params': model.new.parameters(),
    #                       'learning_rate': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.grader.parameters(),
                          'learning_rate': cfg.TRAIN.BASE_LR}]
    
    optimizer = SGD(trainable_params,
                    momentum=cfg.TRAIN.MOMENTUM,
                    weight_decay=L2Decay(cfg.TRAIN.WEIGHT_DECAY))

    lr_scheduler = PiecewiseDecay( boundaries=[50000, 80000],
                                    values=[cfg.TRAIN.BASE_LR, cfg.TRAIN.BASE_LR * 0.1, cfg.TRAIN.BASE_LR * 0.01])
    return optimizer, lr_scheduler

def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model)))
    end = time.time()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                paddle.save(
                        {'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint00_e%d.pdparams' % (epoch))
               
            if epoch == cfg.TRAIN.EPOCH:
                paddle.save( {'epoch': epoch,
                      'state_dict': model.state_dict(),
                       'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint00_efinal.pdparams')
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model, epoch)
                logger.info("model\n{}".format(describe(model)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_lr()
            logger.info('epoch: {}'.format(epoch+1))
                # videorange!!
        if epoch>=cfg.TRAIN.EPOCH//2:
              videorange=cfg.TRAIN.videorangemax
        elif epoch>=cfg.TRAIN.EPOCH//3:
              videorange=cfg.TRAIN.videorangemax-1
        else:
              videorange=cfg.TRAIN.videorangemax-2

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.parameter_list):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['learning_rate']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                          pg['learning_rate'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data,videorange)
        loss = outputs['total_loss'].mean()
        if is_valid_number(loss.item()):
            optimizer.clear_grad()
            loss.backward()
            reduce_gradients(model)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.mean().numpy())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()

def main():
    rank, world_size = dist_init()
    # rank = 0
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                              os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                              logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder_tctrackplus('train')
    dist_model = nn.DataParallel(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../pretrained_models/', cfg.BACKBONE.PRETRAINED)

        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model,
                                            cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model.set_state_dict(paddle.load(cfg.TRAIN.RESUME))
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_path, '../pretrained_models/')
        model.set_state_dict(paddle.load(model_path))
    
    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)

if __name__ == '__main__':
    seed_paddle(args.seed)
    main()
