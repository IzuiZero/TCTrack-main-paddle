import json
from os.path import join, exists
import os
import pandas as pd

dataset_path = 'data'
train_sets = ['GOT-10k_Train_split_01','GOT-10k_Train_split_02','GOT-10k_Train_split_03','GOT-10k_Train_split_04',
            'GOT-10k_Train_split_05','GOT-10k_Train_split_06','GOT-10k_Train_split_07','GOT-10k_Train_split_08',
            'GOT-10k_Train_split_09','GOT-10k_Train_split_10','GOT-10k_Train_split_11','GOT-10k_Train_split_12',
            'GOT-10k_Train_split_13','GOT-10k_Train_split_14','GOT-10k_Train_split_15','GOT-10k_Train_split_16',
            'GOT-10k_Train_split_17','GOT-10k_Train_split_18','GOT-10k_Train_split_19']
val_set = ['val']
d_sets = {'videos_val':val_set,'videos_train':train_sets}


def parse_and_sched(dl_dir='.'):
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            videos = os.listdir(os.path.join(dataset_path,dataset))
            for video in videos:
                if video == 'list.txt':
                    continue
                video = dataset+'/'+video
                gt_path = join(dataset_path, video, 'groundtruth.txt')
                with open(gt_path, 'r') as f:
                    groundtruth = f.readlines()
                for idx, gt_line in enumerate(groundtruth):
                    gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax

                    if video not in js:
                        js[video] = {}
                    if obj not in js[video]:
                        js[video][obj] = {}
                    js[video][obj][frame] = bbox
        if 'videos_val' == d_set:
            with open('val.json', 'w') as json_file:
                json.dump(js, json_file, indent=4, sort_keys=True)
        else:
            with open('train.json', 'w') as json_file:
                json.dump(js, json_file, indent=4, sort_keys=True)
        js = {}

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
