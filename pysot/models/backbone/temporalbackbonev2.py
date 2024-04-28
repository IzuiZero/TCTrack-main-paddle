import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class AttentionLayer(nn.Layer):
    def __init__(self, in_dim):
        super(AttentionLayer, self).__init__()
		
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//16, kernel_size=3, padding=1)
        self.key_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//16, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(axis=-1)
		
    def forward(self, x, y):

        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).reshape(m_batchsize, -1, width*height).transpose((0, 2, 1))
        proj_key = self.key_conv(y).reshape(m_batchsize, -1, width*height)
        energy = paddle.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).reshape(m_batchsize, -1, width*height)

        out = paddle.bmm(proj_value, attention.transpose((0, 2, 1)))
        out = out.reshape(m_batchsize, C, height, width)

        return out
		
		
class CondConv2D(nn.Layer):
    

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.init = nn.Conv2D(in_channels, in_channels, 3, padding=1)
        self.maxpool = nn.AdaptiveMaxPool3D((None, 5, 5))
        self.avgpool = nn.AdaptiveAvgPool3D((None, 1, 1))
        self.temporalconv = nn.Conv3D(in_channels, in_channels, (1, 1, 1))
        self.fc = nn.Conv3D(in_channels, 1, (1, 1, 1))
		
        self.attentionintegrate = AttentionLayer(in_channels)


        self.weight = self.create_parameter(
            shape=[1, 1, out_channels, in_channels // groups, kernel_size, kernel_size],
            default_initializer=paddle.nn.initializer.Constant(value=0))
        if bias:
            self.bias = self.create_parameter(shape=[1, 1, out_channels], default_initializer=paddle.nn.initializer.Constant(value=0))
        else:
            self.bias = None
        
        self.apply(lambda x: nn.initializer.Constant(value=0).init(x.weight) if isinstance(x, nn.Conv3D) else None)
                
		
    def generate_weight(self, xet):
	    
	
        xet = xet.transpose((0, 2, 1, 3, 4))  #x BxCxLxHxW
        xet = self.maxpool(xet)
        
        prior_knowledge = self.init(xet[:,:,0,:,:])
        
        for length in range(xet.shape[2]):
            prior_knowledge = self.attentionintegrate(prior_knowledge.squeeze(2), xet[:,:,length,:,:]).unsqueeze(2)
            
            if length == 0:
                allxet = prior_knowledge
            else:
                allxet = paddle.concat((allxet, prior_knowledge), 2)
        
        
        
        allxet = self.avgpool(allxet) #x BxCxLx1x1
        
        
        calibration = self.temporalconv(allxet)
        
        finalweight = self.weight * (calibration + 1).unsqueeze(0).transpose((1, 3, 0, 2, 4, 5))
		
        bias = self.bias * (self.fc(allxet) + 1).squeeze().unsqueeze(-1)
        
        
        return finalweight, bias, prior_knowledge.squeeze(2)
    
    def initset(self, x):
        finalweight, finalbias, featset = self.generate_weight(x)
        
        b, l, c_in, h, w = x.shape

        x = x.reshape(1, -1, h, w)
        finalweight = finalweight.reshape(-1, self.in_channels, self.kernel_size ,self.kernel_size )
        finalbias = finalbias.reshape(-1)

        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.reshape(-1, self.out_channels, output.shape[-2], output.shape[-1])
        
        
        return output, featset
    
    
    def combinefeat(self, xet, prior_knowledge):
        xet = xet.transpose((0, 2, 1, 3, 4))  #x BxCxLxHxW
        xet = self.maxpool(xet)
        
        prior_knowledge = self.attentionintegrate(prior_knowledge, xet[:,:,0,:,:])
        
        allxet = self.avgpool(prior_knowledge.unsqueeze(2))
        calibration = self.temporalconv(allxet)
        
        finalweight = self.weight * (calibration + 1).unsqueeze(0).transpose((1, 3, 0, 2, 4, 5))
		
        bias = self.bias * (self.fc(allxet) + 1).squeeze().unsqueeze(-1)
        
        

        return finalweight, bias, prior_knowledge
        
    def conti(self, x, feat):
        
        finalweight, finalbias, prior_knowledge = self.combinefeat(x, feat)
        
        b, l, c_in, h, w = x.shape
        
        
        x = x.reshape(1, -1, h, w)
        finalweight = finalweight.reshape(-1, self.in_channels, self.kernel_size ,self.kernel_size )
        finalbias = finalbias.reshape(-1)
	
        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.reshape(-1, self.out_channels, output.shape[-2], output.shape[-1])
       

        return output, prior_knowledge
    
    def forward(self, x): #x B*L*C*W*H
	
        finalweight, finalbias, _ = self.generate_weight(x)
        
        b, l, c_in, h, w = x.shape
        
        
        x = x.reshape(1, -1, h, w)
        finalweight = finalweight.reshape(-1, self.in_channels, self.kernel_size ,self.kernel_size )
        finalbias = finalbias.reshape(-1)
        		
        
        if self.bias is not None:
            
            output = F.conv2d(
                x, weight=finalweight, bias=finalbias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        else:
            output = F.conv2d(
                x, weight=finalweight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=b*l)
        
        output = output.reshape(-1, self.out_channels, output.shape[-2], output.shape[-1])
        return output
		
class TemporalAlexNet(nn.Layer):
    configs = [3, 96, 256, 384, 384, 256]
	
	#input (B*L)*C*W*H, A1,A2,A3,A4,B1,B2,B3,B4...

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), TemporalAlexNet.configs))
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
        self.temporalconv1 = CondConv2D(configs[3], configs[4], kernel_size=3)
        
        self.b_f1=  nn.Sequential(
            nn.BatchNorm2D(configs[4]),
            nn.ReLU(inplace=True))

        self.temporalconv2 = CondConv2D(configs[4], configs[5], kernel_size=3)
        
        self.b_f2= nn.BatchNorm2D(configs[5])
            
        self.feature_size = configs[5]
        self.block1.apply(lambda x: x.requires_grad_(False) if isinstance(x, nn.Conv2D) else None)
        self.block2.apply(lambda x: x.requires_grad_(False) if isinstance(x, nn.Conv2D) else None)
                
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
        		

        
        xset = xset.reshape(-1, xset.shape[-3], xset.shape[-2], xset.shape[-1])
        xset = self.block1(xset)
        xset = self.block2(xset)
        xset = self.block3(xset)
		
        xset = xset.reshape(B, L, xset.shape[-3], xset.shape[-2], xset.shape[-1])
        xset = self.temporalconv1(xset)
        xset = self.b_f1(xset)
        
        
        xset = xset.reshape(B, L, xset.shape[-3], xset.shape[-2], xset.shape[-1])
        xset = self.temporalconv2(xset)
        xset = self.b_f2(xset)
        		
        		
        return xset
