
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channels=None, bias=False):
        super().__init__()

        # If an intermediate channel count isn't provided, match the final width.
        if not mid_channels:
            mid_channels = out_channel

        # Two consecutive Conv-BN-ReLU blocks with 3x3 kernels and padding=1 to keep H, W unchanged.
        self.doubleconv = nn.Sequential(
            # First 3x3 convolution: C_in -> mid_channels
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,          # keep spatial dimensions (same-conv for 3x3)
                bias=bias
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Second 3x3 convolution: mid_channels -> C_out
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,          # keep spatial dimensions (same-conv for 3x3)
                bias=bias
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleconv(x)


class outConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 1×1 conv: channel-wise linear projection, preserves (H, W)
        self.outconv = nn.Conv2d(in_channel, out_channel, kernel_size = 1)

    def forward(self, image):
        return self.outconv(image)


class AdaptivePooling(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.adapool = nn.AdaptiveAvgPool2d(scale)

    def forward(self, feature_map):
        return self.adapool(feature_map)


class PyramidPooling(nn.Module):
    def __init__(self, scales=[1, 2, 3, 6], in_channel=768, out_channel=256, mid_channel=192):
        # here in_channel = 768 (encoder final dim)
        # here out_channel = 768 / 4 = 256.
        # concat channel dim = 768 * 2
        # so after outconv of concat = 768
        super().__init__()
        self.scales = scales  # has to be 1,2,3,6

        for i in range(1, 5):
            setattr(
                self, f"adapool{i}",
                AdaptivePooling(
                    scale=scales[i - 1]
                )
            )
        self.conv = outConv(in_channel=in_channel, out_channel=mid_channel)
        self.bn = nn.BatchNorm2d(mid_channel)

        # TODO use double conv here
        self.ppm_final_conv = DoubleConv(in_channel= 2* in_channel, out_channel=out_channel, mid_channels=in_channel)
    def forward(self, feature_map):
        # first feature map adapool scale 1
        _, _, H, W = feature_map.shape
        upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True)
        pp1 = self.adapool1(feature_map)
        pp1 = self.conv(pp1)
        pp1 = self.bn(pp1)
        pp1 = F.relu(pp1)
        pp1 = upsample(pp1)

        # second feature map adapool scale 2
        pp2 = self.adapool2(feature_map)
        pp2 = self.conv(pp2)
        pp2 = self.bn(pp2)
        pp2 = F.relu(pp2)
        pp2 = upsample(pp2)

        # scale 3
        pp3 = self.adapool3(feature_map)
        pp3 = self.conv(pp3)
        pp3 = self.bn(pp3)
        pp3 = F.relu(pp3)
        pp3 = upsample(pp3)

        # scale 6
        pp4 = self.adapool4(feature_map)
        pp4 = self.conv(pp4)
        pp4 = self.bn(pp4)
        pp4 = F.relu(pp4)
        pp4 = upsample(pp4)

        output = torch.cat((feature_map, pp1, pp2, pp3, pp4), dim=1)
        print(output.shape)

        output = self.ppm_final_conv(output)

        return output


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channel = 768, out_channel = 256):
        super().__init__()
        self.conv = outConv(in_channel=in_channel, out_channel=out_channel)

        # TODO use double conv here
        self.smoothingconv = DoubleConv(in_channel=out_channel, out_channel=out_channel)

    def forward(self, feature_map, previous_feature_map, scale):

        feature_map = self.conv(feature_map)
        feature_map = F.relu(feature_map)
        feature_map = F.interpolate(input=feature_map, scale_factor=scale, mode='bilinear', align_corners=True)


        previous_feature_map = F.interpolate(input=previous_feature_map, scale_factor=2, mode='bilinear', align_corners=True)
        output = feature_map + previous_feature_map
        output = self.smoothingconv(output)
        return output

class Prithvi_EO(nn.Module):
    def __init__(self,
                 pretrained_model=None,
                 num_classes=4,
                 fpn_blocks=[3, 6, 9, 12],
                 scale_factors=[1, 2, 3, 6],
                 embed_dim=768,
                 out_channels_feature_map=256,
                 FPN_out_channels=256,
                 upsampling_scale_list=[4, 2, 1, 0.5],
                 u_height = 56,
                 u_width = 56,
                 ):
        super().__init__()
        self.pretrainedmodel = pretrained_model
        self.upsampling_scales = upsampling_scale_list
        self.scale_factors = scale_factors
        self.fpn_blocks = fpn_blocks
        self.num_classes = num_classes

        self.pyramid_pooling_layer = PyramidPooling(scales=self.scale_factors, in_channel=embed_dim,
                                                    mid_channel=embed_dim // 4, out_channel=out_channels_feature_map)
        self.upsample = nn.Upsample(size = (u_height, u_width), mode='bilinear', align_corners=True)

        self.FPN_upsample = nn.Upsample(scale_factor=4)

        # creating the FPN networks
        self.FPN_block = FeaturePyramidNetwork(in_channel=embed_dim, out_channel=out_channels_feature_map)


        self.FPN_conv = DoubleConv(in_channel=out_channels_feature_map * 4, out_channel=FPN_out_channels, mid_channels=out_channels_feature_map * 2)

        self.segmentation_map_outconv = outConv(in_channel=FPN_out_channels, out_channel=self.num_classes)

    def parse_hidden_output(self, output):
        output = output[:, 1:, :]
        B, patch_dim, C = output.shape
        h = w = int(np.sqrt(patch_dim))  # 14

        output = output.view(B, h, w, C)
        output = output.permute(0, 3, 1, 2)

        return output

    # def feature_map_processing_for_fpn(self, ):
    # TODO mby complete this function

    def forward(self, x):
        enc_output, _, _ = self.pretrainedmodel(x)
        # parse and change view

        hidden_output_3 = self.parse_hidden_output(enc_output[self.fpn_blocks[0] - 1])


        hidden_output_6 = self.parse_hidden_output(enc_output[self.fpn_blocks[1] - 1])

        hidden_output_9 = self.parse_hidden_output(enc_output[self.fpn_blocks[2] - 1])

        final_enc_output = self.parse_hidden_output(enc_output[self.fpn_blocks[3] - 1])
        final_enc_output = F.interpolate(final_enc_output, scale_factor=self.upsampling_scales[3], align_corners=True, mode='bilinear')

        U4 = self.pyramid_pooling_layer(final_enc_output) # 7 7 256

        U3 = self.FPN_block(hidden_output_9, U4, self.upsampling_scales[2])


        U2 = self.FPN_block(hidden_output_6, U3, self.upsampling_scales[1])  # 28 28 256

        U1 = self.FPN_block(hidden_output_3, U2, self.upsampling_scales[0])  # 56 56 256

        # upsample all the Us to the same size which is 56,56
        U2 = self.upsample(U2) # 256 56 56
        U3 = self.upsample(U3) # 256 56 56
        U4 = self.upsample(U4) # 256 56 56

        # concat all these Us
        output = torch.cat((U1, U2, U3, U4), dim=1)  # B 1024 56 56
        output = self.FPN_upsample(output)  # B 1024 224 224
        output = self.FPN_conv(output)  # B 256 224 224

        output = self.segmentation_map_outconv(output)

        return output






