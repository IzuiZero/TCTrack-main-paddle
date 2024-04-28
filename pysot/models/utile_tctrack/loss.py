# Copyright (c) SenseTime. All Rights Reserved.

from paddle import nn
import paddle
import paddle.nn.functional as F


def get_cls_loss(pred, label, select):
    if len(select.shape) == 0 or \
            select.shape == paddle.shape([0]):
        return 0
    pred = paddle.index_select(pred, 0, select)
    label = paddle.index_select(label, 0, select)
    label = paddle.cast(label, 'int64')
    return F.cross_entropy(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.reshape([-1, 2])
    label = label.reshape([-1])
    pos = paddle.nonzero(label.data.eq(1)).squeeze().cuda()
    neg = paddle.nonzero(label.data.eq(0)).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def DISCLE(pred, target, weight):
    pred_x = (pred[:, :, 0] + pred[:, :, 2]) / 2
    pred_y = (pred[:, :, 1] + pred[:, :, 3]) / 2
    pred_w = (-pred[:, :, 0] + pred[:, :, 2])
    pred_h = (-pred[:, :, 1] + pred[:, :, 3])

    target_x = (target[:, :, 0] + target[:, :, 2]) / 2
    target_y = (target[:, :, 1] + target[:, :, 3]) / 2
    target_w = (-target[:, :, 0] + target[:, :, 2])
    target_h = (-target[:, :, 1] + target[:, :, 3])

    loss = paddle.sqrt(paddle.pow((pred_x - target_x), 2) / target_w + paddle.pow((pred_y - target_y), 2) / target_h)

    weight = weight.reshape(loss.shape)

    return (loss * weight).sum() / (weight.sum() + 1e-6)


class IOULoss(nn.Layer):
    def forward(self, pred, target, weight=None):

        pred_left = pred[:, :, 0]
        pred_top = pred[:, :, 1]
        pred_right = pred[:, :, 2]
        pred_bottom = pred[:, :, 3]

        target_left = target[:, :, 0]
        target_top = target[:, :, 1]
        target_right = target[:, :, 2]
        target_bottom = target[:, :, 3]

        target_aera = (target_right - target_left) * \
                      (target_bottom - target_top)
        pred_aera = (pred_right - pred_left) * \
                    (pred_bottom - pred_top)

        w_intersect = paddle.minimum(pred_right, target_right) - paddle.maximum(pred_left, target_left)
        w_intersect = w_intersect.clip(min=0)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) - paddle.maximum(pred_top, target_top)
        h_intersect = h_intersect.clip(min=0)
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = ((area_intersect) / (area_union + 1e-6)).clip(min=0) + 1e-6

        losses = -paddle.log(ious)
        weight = weight.reshape(losses.shape)

        return (losses * weight).sum() / (weight.sum() + 1e-6)
