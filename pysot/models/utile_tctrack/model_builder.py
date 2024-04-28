# Copyright (c) SenseTime. All Rights Reserved.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from pysot.core.config import cfg
from pysot.models.utile_tctrack.loss import select_cross_entropy_loss, IOULoss, DISCLE
from pysot.models.backbone.temporalbackbone import TemporalAlexNet

from pysot.models.utile_tctrack.utile import TCT
from pysot.models.utile_tctrack.utiletest import TCTtest
import numpy as np

class ModelBuilder_tctrack(nn.Layer):
    def __init__(self, label):
        super(ModelBuilder_tctrack, self).__init__()

        self.backbone = TemporalAlexNet()
        self.backbone.eval()

        if label == 'test':
            self.grader = TCTtest(cfg)
        else:
            self.grader = TCT(cfg)

        self.cls3loss = nn.BCEWithLogitsLoss()
        self.IOULOSS = IOULoss()

    def template(self, z, x):
        with paddle.no_grad():
            zf, _, _ = self.backbone.init(z)
            self.zf = zf

            xf, xfeat1, xfeat2 = self.backbone.init(x)

            ppres = self.grader.conv1(self.xcorr_depthwise(xf, zf))

            self.memory = ppres
            self.featset1 = xfeat1
            self.featset2 = xfeat2

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.shape[0]
        channel = kernel.shape[1]
        x = x.reshape([1, batch * channel, x.shape[2], x.shape[3]])
        kernel = kernel.reshape([batch * channel, 1, kernel.shape[2], kernel.shape[3]])
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.reshape([batch, channel, out.shape[2], out.shape[3]])
        return out

    def track(self, x):
        with paddle.no_grad():
            xf, xfeat1, xfeat2 = self.backbone.eachtest(x, self.featset1, self.featset2)

            loc, cls2, cls3, memory = self.grader(xf, self.zf, self.memory)

            self.memory = memory
            self.featset1 = xfeat1
            self.featset2 = xfeat2

        return {
            'cls2': cls2,
            'cls3': cls3,
            'loc': loc
        }

    def log_softmax(self, cls):
        b, a2, h, w = cls.shape
        cls = cls.reshape([b, 2, a2 // 2, h, w])
        cls = cls.transpose([0, 2, 3, 4, 1])
        cls = F.log_softmax(cls, axis=4)

        return cls

    def getcentercuda(self, mapp):
        def dcon(x):
            x = paddle.where(x <= -1, -0.99, x)
            x = paddle.where(x >= 1, 0.99, x)
            return (paddle.log(1 + x) - paddle.log(1 - x)) / 2

        size = mapp.shape[3]

        x = paddle.to_tensor(np.tile((16 * (np.linspace(0, size - 1, size)) + 63) - cfg.TRAIN.SEARCH_SIZE // 2,
                                     size).reshape(-1))
        y = paddle.to_tensor(np.tile((16 * (np.linspace(0, size - 1, size)) + 63).reshape(-1, 1) - cfg.TRAIN.SEARCH_SIZE // 2,
                                     size).reshape(-1))

        shap = dcon(mapp) * (cfg.TRAIN.SEARCH_SIZE // 2)

        xx = np.int16(np.tile(np.linspace(0, size - 1, size), size).reshape(-1))
        yy = np.int16(np.tile(np.linspace(0, size - 1, size).reshape(-1, 1), size).reshape(-1))

        w = shap[:, 0, yy, xx] + shap[:, 1, yy, xx]
        h = shap[:, 2, yy, xx] + shap[:, 3, yy, xx]
        x = x - shap[:, 0, yy, xx] + w / 2 + cfg.TRAIN.SEARCH_SIZE // 2
        y = y - shap[:, 2, yy, xx] + h / 2 + cfg.TRAIN.SEARCH_SIZE // 2

        anchor = paddle.zeros([cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPU, size ** 2, 4])

        anchor[:, :, 0] = x - w / 2
        anchor[:, :, 1] = y - h / 2
        anchor[:, :, 2] = x + w / 2
        anchor[:, :, 3] = y + h / 2

        return anchor

    def forward(self, data):
        presearch = data['pre_search']
        template = data['template']
        search = data['search']
        bbox = data['bbox']
        labelcls2 = data['label_cls2']
        labelxff = data['labelxff']
        labelcls3 = data['labelcls3']
        weightxff = data['weightxff']

        presearch = paddle.concat([presearch, search.unsqueeze(1)], 1)

        zf = self.backbone(template.unsqueeze(1))

        xf = self.backbone(presearch)
        xf = xf.reshape([cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.NUM_GPU, cfg.TRAIN.videorange + 1, xf.shape[2], xf.shape[3], xf.shape[4]])

        loc, cls2, cls3 = self.grader(xf[:, -1, :, :, :], zf, xf[:, :-1, :, :, :].transpose([1, 0, 2, 3, 4]))

        cls2 = self.log_softmax(cls2)

        cls_loss2 = select_cross_entropy_loss(cls2, labelcls2)
        cls_loss3 = self.cls3loss(cls3, labelcls3)

        pre_bbox = self.getcentercuda(loc)
        bbo = self.getcentercuda(labelxff)

        loc_loss1 = self.IOULOSS(pre_bbox, bbo, weightxff)
        loc_loss2 = DISCLE(pre_bbox, bbo, weightxff)
        loc_loss = cfg.TRAIN.w2 * loc_loss1 + cfg.TRAIN.w3 * loc_loss2

        cls_loss = cfg.TRAIN.w4 * cls_loss2 + cfg.TRAIN.w5 * cls_loss3

        outputs = {}
        outputs['total_loss'] = \
            cfg.TRAIN.LOC_WEIGHT * loc_loss \
            + cfg.TRAIN.CLS_WEIGHT * cls_loss

        outputs['cls_loss'] = cls_loss
        outputs['loc_loss1'] = loc_loss1
        outputs['loc_loss2'] = loc_loss2

        return outputs
