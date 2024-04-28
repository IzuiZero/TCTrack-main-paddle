import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn.initializer import Normal, Uniform
from pysot.models.utile_tctrackplus.trantime_paddle import Transformertime


class APN(nn.Layer):

    def __init__(self, cfg):
        super(APN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2D(384, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2D(384, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(256, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2D(256, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )

        channel = 192

        self.convloc = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel // 2, channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel // 4, channel // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 8),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel // 8, 4, kernel_size=3, stride=1, padding=1),
        )

        self.convcls = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel // 2, channel // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel // 4, channel // 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel // 8),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformertime(channel, 6, 1, 2)

        self.cls1 = nn.Conv2D(channel // 8, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2D(channel // 8, 1, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.conv1, self.conv2, self.conv3, self.convloc, self.convcls, self.cls1, self.cls2]:
            for layer in module.sublayers():
                if isinstance(layer, nn.Conv2D):
                    layer.weight.set_value(Uniform(0.0, 0.01))

    def xcorr_depthwise(self, x, kernel):
        batch = kernel.shape[0]
        channel = kernel.shape[1]
        x = x.reshape([1, batch * channel, x.shape[2], x.shape[3]])
        kernel = kernel.reshape([batch * channel, 1, kernel.shape[2], kernel.shape[3]])
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.reshape([batch, channel, out.shape[2], out.shape[3]])
        return out

    def forward(self, x, z, px):
        ppres = self.conv1(self.xcorr_depthwise(px[0], z))

        for i in range(len(px)):
            res3 = self.conv2(self.xcorr_depthwise(px[i], z))
            b, c, w, h = res3.shape
            memory = self.transformer.encoder(res3.reshape([b, c, -1]).transpose([2, 0, 1]),
                                              ppres.reshape([b, c, -1]).transpose([2, 0, 1]))
            ppres = memory.transpose([1, 2, 0]).reshape([b, c, w, h])

        res3 = self.conv2(self.xcorr_depthwise(x, z))
        _, res = self.transformer(res3.reshape([b, c, -1]).transpose([2, 0, 1]),
                                   ppres.reshape([b, c, -1]).transpose([2, 0, 1]),
                                   res3.reshape([b, c, -1]).transpose([2, 0, 1]))
        res = res.transpose([1, 2, 0]).reshape([b, c, w, h])

        loc = self.convloc(res)
        acls = self.convcls(res)

        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)

        return loc, cls1, cls2
