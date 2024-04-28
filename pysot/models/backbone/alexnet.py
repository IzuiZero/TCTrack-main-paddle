import paddle.nn as nn

class AlexNetLegacy(nn.Layer):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else int(x*width_mult), AlexNetLegacy.configs))
        super(AlexNetLegacy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2D(configs[1]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2D(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2D(configs[2]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2D(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2D(configs[3]),
            nn.ReLU(),
            nn.Conv2D(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2D(configs[4]),
            nn.ReLU(),
            nn.Conv2D(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2D(configs[5]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        return x


class AlexNet(nn.Layer):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2D(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2D(configs[1]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2D(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2D(configs[2]),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.ReLU(),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2D(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2D(configs[3]),
            nn.ReLU(),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2D(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2D(configs[4]),
            nn.ReLU(),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2D(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2D(configs[5]),
            )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

def alexnetlegacy(**kwargs):
    return AlexNetLegacy(**kwargs)

def alexnet(**kwargs):
    return AlexNet(**kwargs)
