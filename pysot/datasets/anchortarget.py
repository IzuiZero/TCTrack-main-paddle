import numpy as np
import paddle
from pysot.core.config import cfg
from pysot.utils.bbox import IoU

class AnchorTarget:
    def __init__(self):
        pass

    def select(self, position, keep_num=16):
        num = position.shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return position[slt], keep_num

    def get(self, bbox, size):
        labelcls2 = paddle.zeros((1, size, size))
        pre = (16 * (paddle.linspace(0, size-1, size)) + 63).reshape(-1, 1)
        pr = paddle.zeros((size**2, 2))
        pr[:, 0] = paddle.clip_min(pre.repeat(size, 1).T.reshape(-1), 0)
        pr[:, 1] = paddle.clip_min(pre.repeat(size), 0)
        labelxff = paddle.zeros((4, size, size), dtype='float32')
        labelcls3 = paddle.zeros((1, size, size))
        weightxff = paddle.zeros((1, size, size))

        target = paddle.to_tensor([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        index2 = paddle.clip((target - 63) / 16, min=0, max=size - 1).astype('int32')
        w = index2[2] - index2[0] + 1
        h = index2[3] - index2[1] + 1

        for ii in range(size):
            for jj in range(size):
                weightxff[0, ii, jj] = ((ii - (index2[1] + index2[3]) / 2) / (h / 2))**2 + \
                                       ((jj - (index2[0] + index2[2]) / 2) / (w / 2))**2

        weightxff[weightxff <= 1] = 1 - weightxff[weightxff <= 1]
        weightxff[(weightxff > 1) | (weightxff < 0.5)] = 0

        pos = paddle.nonzero((weightxff.squeeze() < 0.8) & (weightxff.squeeze() >= 0.5))
        num = len(pos)
        pos, _ = self.select(pos, int(num / 1.2))
        weightxff[:, pos[:, 0], pos[:, 1]] = 0

        index = paddle.clip((target - 63) / 16, min=0, max=size - 1).astype('int32')
        w = index[2] - index[0] + 1
        h = index[3] - index[1] + 1

        for ii in range(size):
            for jj in range(size):
                labelcls3[0, ii, jj] = ((ii - (index2[1] + index2[3]) / 2) / (h / 2))**2 + \
                                       ((jj - (index2[0] + index2[2]) / 2) / (w / 2))**2

        labelcls3[labelcls3 <= 1] = 1 - labelcls3[labelcls3 <= 1]
        labelcls3[labelcls3 > 1] = 0

        def con(x):
            return (paddle.exp(x) - paddle.exp(-x)) / (paddle.exp(x) + paddle.exp(-x))

        labelxff[0, :, :] = (pr[:, 0] - target[0]).reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        labelxff[1, :, :] = (target[2] - pr[:, 0]).reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        labelxff[2, :, :] = (pr[:, 1] - target[1]).reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        labelxff[3, :, :] = (target[3] - pr[:, 1]).reshape(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE)
        labelxff = con(labelxff / 143)

        for ii in range(size):
            for jj in range(size):
                labelcls2[0, ii, jj] = ((ii - (index2[1] + index2[3]) / 2) / (h / 2))**2 + \
                                       ((jj - (index2[0] + index2[2]) / 2) / (w / 2))**2

        labelcls2[(labelcls2 > 1)] = -2
        labelcls2[((labelcls2 <= 1) & (labelcls2 >= 0))] = 1 - labelcls2[((labelcls2 <= 1) & (labelcls2 >= 0))]
        labelcls2[((labelcls2 > 0.3) & (labelcls2 < 0.78))] = -1
        labelcls2[((labelcls2 > 0) & (labelcls2 <= 0.3))] = -2

        neg2 = paddle.nonzero(labelcls2.squeeze() == -2)
        neg2, _ = self.select(neg2, int(len(paddle.nonzero(labelcls2 > 0)[0]) * 2))
        labelcls2[:, neg2[:, 0], neg2[:, 1]] = 0

        return labelcls2, labelxff, labelcls3, weightxff
