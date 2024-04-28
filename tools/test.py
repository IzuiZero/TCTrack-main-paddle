# Copyright (c) SenseTime. All Rights Reserved.

import argparse
import os
import cv2
import numpy as np
import paddle
from paddle.vision.models import resnet50

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='TCTrack tracking')
parser.add_argument('--dataset', default='OTB100', type=str,
                    help='datasets')
parser.add_argument('--tracker_name', default='TCTrack', type=str,
                    help='tracker name')
parser.add_argument('--snapshot', default='./tools/snapshot/checkpoint00_e88.pth', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()

def main():
    paddle.set_device('gpu')
    # load config
    if args.tracker_name == "TCTrack":
        if args.dataset in ['UAV123', 'UAV123_10fps', 'DTB70']:
            cfg.merge_from_file(os.path.join('./experiments', args.tracker_name, 'config.yaml'))
        else:
            cfg.merge_from_file(os.path.join('./experiments', args.tracker_name, 'config_l.yaml'))
        # create model
        model = ModelBuilder_tctrack('test')

        # load model
        model = load_pretrain(model, args.snapshot).cuda().eval()

        # build tracker
        tracker = TCTrackTracker(model)
        hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    elif args.tracker_name == "TCTrack++":
        cfg.merge_from_file(os.path.join('./experiments', args.tracker_name, 'config.yaml'))
        # create model
        model = ModelBuilder_tctrackplus('test')

        # load model
        model = load_pretrain(model, args.snapshot).cuda().eval()

        # build tracker
        tracker = TCTrackplusTracker(model)
        hp = getattr(cfg.HP_SEARCH_TCTrackpp_offline, args.dataset)

    else:
        print('No such tracker')

    cur_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_root = os.path.join(cur_dir, '../test_dataset', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.tracker_name

    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img, hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

if __name__ == '__main__':
    main()
