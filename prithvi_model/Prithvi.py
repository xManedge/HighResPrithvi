import os
os.environ['TORCH_DYNAMO_DISABLE_DOCSTRING_CHECKS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prithvi_model.modules import (
    DoubleConv,
    outConv,
    PyramidPooling,
    FeaturePyramidNetwork,

)

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
        enc_output = self.pretrainedmodel(x)
        # parse and change view

        hidden_output_3 = self.parse_hidden_output(enc_output[self.fpn_blocks[0] - 1]) # block 3


        hidden_output_6 = self.parse_hidden_output(enc_output[self.fpn_blocks[1] - 1]) # block 6

        hidden_output_9 = self.parse_hidden_output(enc_output[self.fpn_blocks[2] - 1]) # block 9

        final_enc_output = self.parse_hidden_output(enc_output[self.fpn_blocks[3] - 1]) # block 12 - final block
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