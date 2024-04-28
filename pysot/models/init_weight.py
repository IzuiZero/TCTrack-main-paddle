import paddle.nn as nn

def init_weights(model):
    for m in model.sublayers():
        if isinstance(m, nn.Conv2D):
            nn.initializer.KaimingNormal(0.0, mode='fan_out', nonlinearity='relu')(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(value=0.0)(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            nn.initializer.Constant(value=1.0)(m.weight)
            nn.initializer.Constant(value=0.0)(m.bias)
