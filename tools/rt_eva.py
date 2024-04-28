import argparse
import pickle
from os.path import join, isfile
import numpy as np
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='./tools/results_rt_raw/OTB100/', type=str,
                        help='raw result root')
    parser.add_argument('--tar_root', default='./tools/results_rt/OTB100', type=str,
                        help='target result root')
    parser.add_argument('--gtroot', default='./test_dataset/OTB100', type=str)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    trackers = os.listdir(args.raw_root)
    gt_path = args.gtroot
    if 'DTB70' in gt_path:
        seqs = os.listdir(gt_path)
        gt_list = []
        for seq in seqs:
            gt_list.append(os.path.join(gt_path, seq, 'groundtruth_rect.txt'))
    elif 'OTB' in gt_path:
        seqs = os.listdir(gt_path)
        gt_list = []
        for seq in seqs:
            if seq == 'CVPR13.json' or seq == 'OTB50.json' or seq == 'OTB100.json':
                continue
            gt_list.append(os.path.join(gt_path, seq, 'groundtruth_rect.txt'))
    else:
        gt_list = os.listdir(gt_path)
        gt_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path) if i.endswith('.txt')]
    
    for tracker in trackers:        
        ra_path = join(args.raw_root, tracker)
        ou_path = join(args.tar_root, tracker)
        if os.path.isdir(ou_path):
            continue
        mismatch = 0
        fps_a = []
        for gt_idx, video in enumerate(gt_list):
            name = video.split('/')[-1][0:-4]
            name_rt = name[0:-3]
            if 'DTB70' in gt_path:
                name = video.split('/')[-2]
                name_rt = name
            elif 'OTB100' in gt_path:
                name = video.split('/')[-2]
                name_rt = name
            print('Pairing {:s} output with the ground truth ({:d}/{:d}): {:s}'.format(tracker, len(gt_list), gt_idx, name))
            results = pickle.load(open(join(ra_path, name + '.pkl'), 'rb'))
            gtlen = len(open(join(video)).readlines())
            results_raw = results.get('results_raw', None)
            timestamps = results['timestamps']
            timestamps[0] = 0
            input_fidx = results['input_fidx']
            run_time = results['runtime']
            fps_a.append(len(run_time) / sum(run_time))
            tidx_p1 = 0
            pred_bboxes = []
            for idx in range(gtlen):
                t = (idx - args.eta) / args.fps
                while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
                    tidx_p1 += 1
                tidx = tidx_p1 - 1
                ifidx = input_fidx[tidx]
                mismatch += idx - ifidx
                pred_bboxes.append(results_raw[tidx])
            if not os.path.isdir(ou_path):
                os.makedirs(ou_path)
            result_path = join(ou_path, '{}.txt'.format(name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
        fps_path = join(ou_path, '{}.txt'.format('Speed'))
        with open(fps_path, 'w') as f:
            f.write(str(sum(fps_a) / len(fps_a)))

if __name__ == '__main__':
    main()
