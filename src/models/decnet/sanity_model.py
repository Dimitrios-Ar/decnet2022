import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models.resnet import resnet101

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

        return rgbd_output