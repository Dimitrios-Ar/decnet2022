import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models.resnet import resnet101

def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

class I2D(nn.Module):
    def __init__(self, pretrained=False, fixed_feature_weights=True):
        super(I2D, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)
        self.agg4 = agg_node(256, 128)
        
        # Upshuffle layers
        self.up1 = upshuffle(128,128,8)
        self.up2 = upshuffle(128,128,4)
        self.up3 = upshuffle(128,128,2)
        
        # Depth prediction
        self.predict1 = smooth(512, 128)
        self.predict2 = predict(128, 1)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        _,_,H,W = x.size()
        
        # Bottom-up
        print(x.shape)
        c1 = self.layer0(x)
        print(c1.shape)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)
        
        # Top-down predict and refine
        d5, d4, d3, d2 = self.up1(self.agg1(p5)), self.up2(self.agg2(p4)), self.up3(self.agg3(p3)), self.agg4(p2)
        _,_,H,W = d2.size()
        vol = torch.cat( [ F.interpolate(d, size=(H,W), mode='bilinear', align_corners=True) for d in [d5,d4,d3,d2] ], dim=1 )
        
        # return self.predict2( self.up4(self.predict1(vol)) )
        return self.predict2( self.predict1(vol) )     # img : depth = 4 : 1 

print("Imports passed")

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)


class DecNet(nn.Module):
    def __init__(self, input_size = ()):
        super(DecNet, self).__init__()
        #self.conv3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet = resnet101(pretrained=False)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        #print('x.size',x.size())
        _,_,H,W = x.size()
        #print(x.shape)
        # Bottom-up
        prints = True
        c1 = self.conv1(x)
        c2 = self.bn1(c1)
        c3 = self.maxpool(c2)
        c4 = F.relu(c3)
        c5 = self.conv2(c4)
        c6 = F.relu(c5)
        c7 = self.conv3(c6)
        c8 = F.relu(c7)

        if prints == True:
            print('x', x.shape)
            print('c1',c1.shape)
            print('c2',c2.shape)
            print('c3',c3.shape)
            print('c4',c4.shape)
            print('c5',c5.shape)
            print('c6', c6.shape)
            print('c7',c7.shape)
            print('c8', c8.shape)

        return c8

class DecNetRGB(nn.Module):
    def __init__(self, input_size = ()):
        super(DecNetRGB, self).__init__()

        self.rgb_encoder_init = convbnrelu(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer0 = convbnrelu(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer1 = convbnrelu(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer2 = convbnrelu(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        #self.rgb_encoder_layer3 = convbnrelu(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        

        #self.rgb_decoder_layer3 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer1 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)


    def forward(self, data):
        #print('x.size',x.size())

        #b 1 352 1216
        rgb = data

        rgb_feature_init = self.rgb_encoder_init(rgb) # SIZE
        rgb_feature_0 = self.rgb_encoder_layer0(rgb_feature_init) # SIZE
        rgb_feature_1 = self.rgb_encoder_layer1(rgb_feature_0) # SIZE
        #rgb_feature_2 = self.rgb_encoder_layer2(rgb_feature_1) # SIZE
        #rgb_feature_3 = self.rgb_encoder_layer3(rgb_feature_2) # SIZE

        #rgb_dec_feature_3 = self.rgb_decoder_layer3(rgb_feature_2) # SIZE
        #rgb_dec_feature_2 = self.rgb_decoder_layer2(rgb_feature_2) # SIZE
        rgb_dec_feature_1 = self.rgb_decoder_layer1(rgb_feature_1) # SIZE
        rgb_dec_feature_0 = self.rgb_decoder_layer0(rgb_dec_feature_1) # SIZE
        
        rgb_output = self.rgb_decoder_output(rgb_dec_feature_0)






        prints = False
        if prints == True:
            print('rgb', rgb.shape)#[8, 3, 232, 304])
            print('d', d.shape)#[8, 1, 232, 304])
            print('rgb_feature_init',rgb_feature_init.shape)#[8, 32, 116, 152])
            print('rgb_feature_0',rgb_feature_0.shape)#[8, 64, 58, 76])
            print('rgb_feature_1',rgb_feature_1.shape)#[[8, 128, 29, 38])] 
            #print('rgb_feature_2',rgb_feature_2.shape)
            #print('rgb_dec_feature_2',rgb_dec_feature_2.shape)

            print('rgb_dec_feature_1',rgb_dec_feature_1.shape)#[8, 64, 58, 76])
            print('rgb_dec_feature_0',rgb_dec_feature_0.shape)#[8, 32, 116, 152])
            print('rgb_output', rgb_output.shape)#[8, 1, 116, 152])
            #print('c7',c7.shape)
            #print('c8', c8.shape)

        return rgb_output

class DecNetRGBD(nn.Module):
    def __init__(self, input_size = ()):
        super(DecNetRGBD, self).__init__()

        self.rgb_encoder_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer0 = convbnrelu(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer1 = convbnrelu(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer2 = convbnrelu(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        #self.rgb_encoder_layer3 = convbnrelu(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        

        #self.rgb_decoder_layer3 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer1 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)

        

    def forward(self, rgb, d):
        #print('x.size',x.size())

        #b 1 352 1216
        rgb_feature = torch.cat((rgb, d), dim=1)
        #rgb_feature = rgb

 


        rgb_feature_init = self.rgb_encoder_init(rgb_feature) # SIZE
        rgb_feature_0 = self.rgb_encoder_layer0(rgb_feature_init) # SIZE
        rgb_feature_1 = self.rgb_encoder_layer1(rgb_feature_0) # SIZE
        #rgb_feature_2 = self.rgb_encoder_layer2(rgb_feature_1) # SIZE
        #rgb_feature_3 = self.rgb_encoder_layer3(rgb_feature_2) # SIZE

        #rgb_dec_feature_3 = self.rgb_decoder_layer3(rgb_feature_2) # SIZE
        #rgb_dec_feature_2 = self.rgb_decoder_layer2(rgb_feature_2) # SIZE
        rgb_dec_feature_1 = self.rgb_decoder_layer1(rgb_feature_1) # SIZE
        rgb_dec_feature_0 = self.rgb_decoder_layer0(rgb_dec_feature_1) # SIZE
        
        rgb_output = self.rgb_decoder_output(rgb_dec_feature_0)






        prints = True
        if prints == True:
            print('rgb', rgb.shape)#[8, 3, 232, 304])
            print('d', d.shape)#[8, 1, 232, 304])
            print('rgb_feature_init',rgb_feature_init.shape)#[8, 32, 116, 152])
            print('rgb_feature_0',rgb_feature_0.shape)#[8, 64, 58, 76])
            print('rgb_feature_1',rgb_feature_1.shape)#[[8, 128, 29, 38])] 
            #print('rgb_feature_2',rgb_feature_2.shape)
            #print('rgb_dec_feature_2',rgb_dec_feature_2.shape)

            print('rgb_dec_feature_1',rgb_dec_feature_1.shape)#[8, 64, 58, 76])
            print('rgb_dec_feature_0',rgb_dec_feature_0.shape)#[8, 32, 116, 152])
            print('rgb_output', rgb_output.shape)#[8, 1, 116, 152])
            #print('c7',c7.shape)
            #print('c8', c8.shape)


        return rgb_output

class DecNetRGBDsmall(nn.Module):
    def __init__(self, input_size = ()):
        super(DecNetRGBDsmall, self).__init__()

        self.rgb_encoder_init = convbnrelu(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer0 = convbnrelu(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.rgb_encoder_layer1 = convbnrelu(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        #self.rgb_encoder_layer2 = convbnrelu(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        #self.rgb_encoder_layer3 = convbnrelu(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        

        #self.rgb_decoder_layer3 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgbd_decoder_layer2 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgbd_decoder_layer1 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgbd_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgbd_decoder_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0)

        self.d_encoder_init = convbnrelu(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.d_encoder_layer0 = convbnrelu(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.d_encoder_layer1 = convbnrelu(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        

    def forward(self, rgb, d):
        #print('x.size',x.size())

        #b 1 352 1216
        #rgb_feature = torch.cat((rgb, d), dim=1)
        rgb_feature = rgb
 
        d_feature_init = self.d_encoder_init(d) # SIZE
        d_feature_0 = self.d_encoder_layer0(d_feature_init) # SIZE
        d_feature_1 = self.d_encoder_layer1(d_feature_0) # SIZE

        rgb_feature_init = self.rgb_encoder_init(rgb_feature) # SIZE
        rgb_feature_0 = self.rgb_encoder_layer0(rgb_feature_init) # SIZE
        rgb_feature_1 = self.rgb_encoder_layer1(rgb_feature_0) # SIZE
        #rgb_feature_2 = self.rgb_encoder_layer2(rgb_feature_1) # SIZE
        #rgb_feature_3 = self.rgb_encoder_layer3(rgb_feature_2) # SIZE

        #rgb_dec_feature_3 = self.rgb_decoder_layer3(rgb_feature_2) # SIZE
        rgbd_dec_feature_2 = self.rgbd_decoder_layer2(torch.cat((rgb_feature_1, d_feature_1), dim=1)) # SIZE
        rgbd_dec_feature_1 = self.rgbd_decoder_layer1(rgbd_dec_feature_2) # SIZE
        rgbd_dec_feature_0 = self.rgbd_decoder_layer0(rgbd_dec_feature_1) # SIZE
        
        rgbd_output = self.rgbd_decoder_output(rgbd_dec_feature_0)
        #print('rgb_output', rgbd_output.shape)#[8, 1, 116, 152])






        prints = False
        if prints == True:
            print('rgb', rgb.shape)#[8, 3, 232, 304])
            print('d', d.shape)#[8, 1, 232, 304])
            print('rgb_feature_init',rgb_feature_init.shape)#[8, 32, 116, 152])
            print('rgb_feature_0',rgb_feature_0.shape)#[8, 64, 58, 76])
            print('rgb_feature_1',rgb_feature_1.shape)#[[8, 128, 29, 38])] 
            #print('rgb_feature_2',rgb_feature_2.shape)
            #print('rgb_dec_feature_2',rgb_dec_feature_2.shape)

            print('rgb_dec_feature_2',rgbd_dec_feature_2.shape)#[8, 64, 58, 76])
            print('rgb_dec_feature_1',rgbd_dec_feature_1.shape)#[8, 32, 116, 152])
            print('rgb_dec_feature_0',rgbd_dec_feature_0.shape)#[8, 32, 116, 152])
            print('rgb_output', rgbd_output.shape)#[8, 1, 116, 152])
            #print('c7',c7.shape)
            #print('c8', c8.shape)

        return rgbd_output