import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CondConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.avgpool = nn.AdaptiveAvgPool3D((None, 1, 1))
        self.temporalconv = nn.Conv3D(in_channels, in_channels, (3, 1, 1))
        self.fc = nn.Conv3D(in_channels, 1, (3, 1, 1))

        self.weight = self.create_parameter(
            shape=[1, 1, out_channels, in_channels // groups, kernel_size, kernel_size],
            default_initializer=nn.initializer.KaimingNormal())
        if bias:
            self.bias = self.create_parameter(shape=[1, 1, out_channels],
                                               default_initializer=nn.initializer.Constant(0))
        else:
            self.bias = None

        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv3D):
                sublayer.weight.set_value(paddle.zeros_like(sublayer.weight))
                sublayer.bias.set_value(paddle.zeros_like(sublayer.bias))

    def generateweight(self, xet):
        xet = xet.transpose((0, 2, 1, 3, 4))  # x BxCxLxHxW
        xet = self.avgpool(xet)  # x BxCxLx1x1

        allxet = paddle.concat((xet[:, :, 0, :, :].unsqueeze(2), xet[:, :, 0, :, :].unsqueeze(2), xet), 2)
        calibration = self.temporalconv(allxet)

        finalweight = self.weight * (calibration + 1).unsqueeze(0).transpose((1, 3, 0, 2, 4, 5))

        bias = self.bias * (self.fc(allxet) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias, allxet

    def initset(self, x):
        finalweight, finalbias, featset = self.generateweight(x)

        b, l, c_in, h, w = x.shape

        x = x.reshape([1, -1, h, w])
        finalweight = finalweight.reshape([-1, self.in_channels, self.kernel_size, self.kernel_size])
        finalbias = finalbias.reshape([-1])

        if self.bias is not None:
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)

        output = output.reshape([-1, self.out_channels, output.shape[-2], output.shape[-1]])

        return output, featset

    def combinefeat(self, xet, feat):
        xet = xet.transpose((0, 2, 1, 3, 4))  # x BxCxLxHxW
        xet = self.avgpool(xet)  # x BxCxLx1x1

        allxet = paddle.concat((feat[:, :, -2, :, :].unsqueeze(2), feat[:, :, -1, :, :].unsqueeze(2), xet), 2)
        calibration = self.temporalconv(allxet)

        finalweight = self.weight * (calibration + 1).unsqueeze(0).transpose((1, 3, 0, 2, 4, 5))

        bias = self.bias * (self.fc(allxet) + 1).squeeze().unsqueeze(-1)

        return finalweight, bias, allxet

    def conti(self, x, feat):

        finalweight, finalbias, allxet = self.combinefeat(x, feat)

        b, l, c_in, h, w = x.shape

        x = x.reshape([1, -1, h, w])
        finalweight = finalweight.reshape([-1, self.in_channels, self.kernel_size, self.kernel_size])
        finalbias = finalbias.reshape([-1])

        if self.bias is not None:
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)

        output = output.reshape([-1, self.out_channels, output.shape[-2], output.shape[-1]])

        return output, allxet

    def forward(self, x):
        finalweight, finalbias, _ = self.generateweight(x)

        b, l, c_in, h, w = x.shape

        x = x.reshape([1, -1, h, w])
        finalweight = finalweight.reshape([-1, self.in_channels, self.kernel_size, self.kernel_size])
        finalbias = finalbias.reshape([-1])

        if self.bias is not None:
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b * l)

        output = output.reshape([-1, self.out_channels, output.shape[-2], output.shape[-1]])
        return output

class TemporalAlexNet(nn.Layer):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else int(x * width_mult), TemporalAlexNet.configs))
        super(TemporalAlexNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2D(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2D(configs[1]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2D(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2D(configs[2]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2D(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2D(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.temporalconv1 = CondConv2d(configs[3], configs[4], kernel_size=3)

        self.b_f1 = nn.Sequential(
            nn.BatchNorm2D(configs[4]),
            nn.ReLU(inplace=True))

        self.temporalconv2 = CondConv2d(configs[4], configs[5], kernel_size=3)

        self.b_f2 = nn.BatchNorm2D(configs[5])
        
        for layer in [self.block1, self.block2]:
            for param in layer.parameters():
                param.stop_gradient = True

    def init(self, xset):
        xset = self.block1(xset)
        xset = self.block2(xset)
        xset = self.block3(xset)

        xset = paddle.unsqueeze(xset, 1)
        xset, feat1 = self.temporalconv1.initset(xset)

        xset = self.b_f1(xset)

        xset = paddle.unsqueeze(xset, 1)
        xset, feat2 = self.temporalconv2.initset(xset)

        xset = self.b_f2(xset)

        return xset, feat1, feat2

    def eachtest(self, xset, feat1, feat2):
        xset = self.block1(xset)
        xset = self.block2(xset)
        xset = self.block3(xset)

        xset = paddle.unsqueeze(xset, 1)
        xset, feat1 = self.temporalconv1.conti(xset, feat1)
        xset = self.b_f1(xset)

        xset = paddle.unsqueeze(xset, 1)
        xset, feat2 = self.temporalconv2.conti(xset, feat2)
        xset = self.b_f2(xset)

        return xset, feat1, feat2

    def forward(self, xset):
        B, L, _, _, _ = xset.shape

        xset = xset.reshape([-1, xset.shape[-3], xset.shape[-2], xset.shape[-1]])
        xset = self.block1(xset)
        xset = self.block2(xset)
        xset = self.block3(xset)

        xset = xset.reshape([B, L, xset.shape[-3], xset.shape[-2], xset.shape[-1]])
        xset = self.temporalconv1(xset)
        xset = self.b_f1(xset)

        xset = xset.reshape([B, L, xset.shape[-3], xset.shape[-2], xset.shape[-1]])
        xset = self.temporalconv2(xset)
        xset = self.b_f2(xset)

        return xset
