# Copyright (c) SenseTime. All Rights Reserved.

import paddle
import paddle.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.shape[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.unsqueeze(0)
        pk = pk.unsqueeze(0)
        po = F.conv2d(px, pk)
        out.append(po)
    out = paddle.concat(out, axis=0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.shape[0]
    pk = kernel.view(-1, x.shape[1], kernel.shape[2], kernel.shape[3])
    px = x.view(1, -1, x.shape[2], x.shape[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.shape[2], po.shape[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.shape[0]
    channel = kernel.shape[1]
    x = x.view(1, batch*channel, x.shape[2], x.shape[3])
    kernel = kernel.view(batch*channel, 1, kernel.shape[2], kernel.shape[3])
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.shape[2], out.shape[3])
    return out
