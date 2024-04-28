import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from pysot.models.utile_tctrack.trantime import Transformertime

class TCTtest(nn.Layer):
    
    def __init__(self,cfg):
        super(TCTtest, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2D(256, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(256, 192, kernel_size=3, bias_attr=False, stride=2, padding=1),
            nn.BatchNorm2D(192),
            nn.ReLU(inplace=True),
        )
        
        channel = 192

        self.convloc = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, 4, kernel_size=3, stride=1, padding=1),
        )
        
        self.convcls = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformertime(channel, 6, 1, 2)
        self.cls1 = nn.Conv2D(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2D(channel, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.conv1, self.conv2, self.convloc, self.convcls, self.cls1, self.cls2]:
            for l in modules.sublayers():
                if isinstance(l, nn.Conv2D):
                    l.weight.set_value(paddle.normal(mean=0.0, std=0.01))

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation"""
        batch = kernel.shape[0]
        channel = kernel.shape[1]
        x = x.reshape([1, batch*channel, x.shape[2], x.shape[3]])
        kernel = kernel.reshape([batch*channel, 1, kernel.shape[2], kernel.shape[3]])
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.reshape([batch, channel, out.shape[2], out.shape[3]])
        return out
    
    def forward(self, x, z, ppres):
        
        res3 = self.conv2(self.xcorr_depthwise(x, z))
    
        b, c, w, h = res3.shape
        memory, res = self.transformer(res3.reshape([b, c, -1]).transpose([2, 0, 1]), \
                                       ppres.reshape([b, c, -1]).transpose([2, 0, 1]), \
                                       res3.reshape([b, c, -1]).transpose([2, 0, 1]))
        res = res.transpose([1, 2, 0]).reshape([b, c, w, h])
        
        loc = self.convloc(res)
        acls = self.convcls(res)

        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)

        return loc, cls1, cls2, memory
