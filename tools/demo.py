import os
import cv2
import paddle
from glob import glob
from paddle.vision.models import resnet50

from pysot.core.config import cfg
from pysot.models.utile.model_builder import ModelBuilder
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

paddle.set_device('cpu')

# Mock argparse.ArgumentParser for demo
class Args:
    def __init__(self):
        self.config = '../experiments/TCTrack/config.yaml'
        self.snapshot = './snapshot/tctrack.pdparams'
        self.video_name = '../test_dataset/sequence_name'

args = Args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = paddle.device.is_compiled_with_cuda()
    device = paddle.device.get_device()

    # create model
    model = ModelBuilder('test')

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = TCTrackTracker(model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)

if __name__ == '__main__':
    main()
