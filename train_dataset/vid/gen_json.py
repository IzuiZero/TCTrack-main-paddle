from os.path import join
import json
import numpy as np

print('load json (raw vid info), please wait 20 seconds~')
vid = json.load(open('vid.json', 'r'))


def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok


snippets = dict()
n_snippets = 0
n_videos = 0
for subset in vid:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        id_set = []
        id_frames = [[]] * 60  # at most 60 objects
        for f, frame in enumerate(frames):
            objs = frame['objs']
            frame_sz = frame['frame_sz']
            for obj in objs:
                trackid = obj['trackid']
                occluded = obj['occ']
                bbox = obj['bbox']

                if trackid not in id_set:
                    id_set.append(trackid)
                    id_frames[trackid] = []
                id_frames[trackid].append(f)
        if len(id_set) > 0:
            snippets[video['base_path']] = dict()
        for selected in id_set:
            frame_ids = sorted(id_frames[selected])
            sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
            sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
            for seq in sequences:
                snippet = dict()
                for frame_id in seq:
                    frame = frames[frame_id]
                    for obj in frame['objs']:
                        if obj['trackid'] == selected:
                            o = obj
                            continue
                    snippet[frame['img_path'].split('.')[0]] = o['bbox']
                snippets[video['base_path']]['{:02d}'.format(selected)] = snippet
                n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))
        
train = {k:v for (k,v) in snippets.items() if 'train' in k}
val = {k:v for (k,v) in snippets.items() if 'val' in k}

with open('train.json', 'w') as train_file:
    json.dump(train, train_file, indent=4, sort_keys=True)

with open('val.json', 'w') as val_file:
    json.dump(val, val_file, indent=4, sort_keys=True)

print('done!')
