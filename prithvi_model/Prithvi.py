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


"""
===============================================================================
Prithvi_EO: Earth Observation Segmentation Model
===============================================================================

This class defines the full segmentation network architecture built on top of
a pretrained encoder (such as Prithvi or ViT-based backbones).

The model combines:
    - A frozen pretrained encoder for feature extraction.
    - A Pyramid Pooling module for multi-scale contextual aggregation.
    - A Feature Pyramid Network (FPN) for multi-level feature fusion.
    - Upsampling and concatenation stages for dense prediction.
    - A final 1×1 convolution head for class segmentation output.

The design follows the general UNet-FPN philosophy but tailored to transformer
encoder outputs.

Mathematically:
Let f_i ∈ ℝ^{B × C × H_i × W_i} denote intermediate feature maps from encoder.
Then:
    U4 = PyramidPooling(f_12)
    U3 = FPN(f_9, U4)
    U2 = FPN(f_6, U3)
    U1 = FPN(f_3, U2)
Output segmentation map:
    Ŷ = Conv1×1(Concat(U1, U2↑, U3↑, U4↑))

Each stage ensures spatial alignment via bilinear interpolation.
===============================================================================
"""
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
                 u_height=56,
                 u_width=56,
                 ):
        super().__init__()
        """ 
        ----------------------------
        MODEL INITIALIZATION
        ----------------------------
        """

        # Pretrained transformer/encoder backbone (frozen)
        self.pretrainedmodel = pretrained_model

        # Store parameters for feature scaling and FPN construction
        self.upsampling_scales = upsampling_scale_list
        self.scale_factors = scale_factors
        self.fpn_blocks = fpn_blocks
        self.num_classes = num_classes

        # ---------------------------------------------------------------------
        # Pyramid Pooling: used at final encoder output to integrate multi-scale
        # spatial context from deep features.
        # Input: (B, embed_dim, H, W)
        # Output: (B, out_channels_feature_map, H, W)
        # ---------------------------------------------------------------------
        self.pyramid_pooling_layer = PyramidPooling(
            scales=self.scale_factors,
            in_channel=embed_dim,
            mid_channel=embed_dim // 4,
            out_channel=out_channels_feature_map
        )

        # Upsample layer to bring all FPN outputs to a uniform resolution (e.g., 56×56)
        self.upsample = nn.Upsample(size=(u_height, u_width), mode='bilinear', align_corners=True)

        # Final upsample to bring concatenated FPN output to target resolution (224×224)
        self.FPN_upsample = nn.Upsample(scale_factor=4)

        # Single FPN block used iteratively to fuse hierarchical encoder features
        self.FPN_block = FeaturePyramidNetwork(in_channel=embed_dim, out_channel=out_channels_feature_map)

        # After concatenating all U1..U4 (4×256=1024 channels),
        # use DoubleConv to compress to FPN_out_channels (e.g., 256)
        self.FPN_conv = DoubleConv(
            in_channel=out_channels_feature_map * 4,
            out_channel=FPN_out_channels,
            mid_channels=out_channels_feature_map * 2
        )

        # Final segmentation head: 1×1 conv to map to num_classes
        self.segmentation_map_outconv = outConv(
            in_channel=FPN_out_channels,
            out_channel=self.num_classes
        )

        # Freeze pretrained encoder weights
        for param in self.pretrainedmodel.parameters():
            param.requires_grad = False


    """
    ----------------------------------------------------------------------------
    parse_hidden_output:
    ----------------------------------------------------------------------------
    Converts a transformer patch embedding sequence into a 2D feature map.

    Input:
        output: Tensor of shape (B, N+1, C)
            - N: number of patches (H×W)
            - C: embedding dimension
            - index 0 corresponds to CLS token (removed)

    Steps:
        1. Remove CLS token → (B, N, C)
        2. Compute spatial grid size h = w = √N
        3. Reshape → (B, C, h, w)
    """
    def parse_hidden_output(self, output):
        # Drop the CLS token from the first embedding position
        output = output[:, 1:, :]

        # Extract batch, patch count, and channel dims
        B, patch_dim, C = output.shape

        # Compute patch grid dimensions (assume square layout)
        h = w = int(np.sqrt(patch_dim))  # e.g., 14×14 grid from 196 patches

        # Reshape the flat sequence of patches into 2D feature maps
        output = output.view(B, h, w, C)  # (B, H, W, C)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)

        return output


    """
    ----------------------------------------------------------------------------
    forward:
    ----------------------------------------------------------------------------
    Forward pass of the segmentation model.

    Steps:
        1. Pass input through pretrained encoder to extract multi-level features.
        2. Parse patch embeddings into spatial feature maps.
        3. Apply pyramid pooling on deepest feature map.
        4. Sequentially fuse features via FPN (U4 → U3 → U2 → U1).
        5. Upsample and concatenate all U maps.
        6. Compress concatenated tensor and project to segmentation logits.

    Tensor shapes (example for ViT-Large input 224×224):
        hidden_output_3 : (B, 768, 14, 14)
        hidden_output_6 : (B, 768, 14, 14)
        hidden_output_9 : (B, 768, 14, 14)
        final_enc_output: (B, 768, 14, 14)
        After upsampling:
            U1..U4 → (B, 256, 56, 56)
        Concatenation:
            (B, 1024, 56, 56)
        Final output:
            (B, num_classes, 224, 224)
    """
    def forward(self, x):
        # ---------------------------------------------------------------------
        # STEP 1: Encoder forward pass
        # ---------------------------------------------------------------------
        enc_output = self.pretrainedmodel(x)  # list of hidden states per block

        # ---------------------------------------------------------------------
        # STEP 2: Parse intermediate transformer features into 2D feature maps
        # ---------------------------------------------------------------------
        hidden_output_3 = self.parse_hidden_output(enc_output[self.fpn_blocks[0] - 1])  # Block 3 features
        hidden_output_6 = self.parse_hidden_output(enc_output[self.fpn_blocks[1] - 1])  # Block 6 features
        hidden_output_9 = self.parse_hidden_output(enc_output[self.fpn_blocks[2] - 1])  # Block 9 features
        final_enc_output = self.parse_hidden_output(enc_output[self.fpn_blocks[3] - 1])  # Block 12 (deepest layer)

        # ---------------------------------------------------------------------
        # STEP 3: Pyramid pooling for multi-scale context aggregation
        # ---------------------------------------------------------------------
        final_enc_output = F.interpolate(
            final_enc_output,
            scale_factor=self.upsampling_scales[3],
            align_corners=True,
            mode='bilinear'
        )
        U4 = self.pyramid_pooling_layer(final_enc_output)  # Deepest feature (7×7 → contextualized 256×7×7)

        # ---------------------------------------------------------------------
        # STEP 4: Top-down FPN refinement (hierarchical feature fusion)
        # Each U map incorporates higher-resolution features progressively.
        # ---------------------------------------------------------------------
        U3 = self.FPN_block(hidden_output_9, U4, self.upsampling_scales[2])
        U2 = self.FPN_block(hidden_output_6, U3, self.upsampling_scales[1])
        U1 = self.FPN_block(hidden_output_3, U2, self.upsampling_scales[0])

        # ---------------------------------------------------------------------
        # STEP 5: Normalize spatial resolutions (upsample all to 56×56)
        # ---------------------------------------------------------------------
        U2 = self.upsample(U2)  # Align (B,256,56,56)
        U3 = self.upsample(U3)
        U4 = self.upsample(U4)

        # ---------------------------------------------------------------------
        # STEP 6: Concatenate multi-level feature maps and compress
        # ---------------------------------------------------------------------
        output = torch.cat((U1, U2, U3, U4), dim=1)  # Channel concat: 4×256 = 1024 channels
        output = self.FPN_upsample(output)            # Upsample to (224×224)
        output = self.FPN_conv(output)                # Spatial fusion via DoubleConv

        # ---------------------------------------------------------------------
        # STEP 7: Segmentation prediction head
        # ---------------------------------------------------------------------
        output = self.segmentation_map_outconv(output)  # (B, num_classes, H, W)

        # Output is dense per-pixel class logits
        return output


    """
    ----------------------------------------------------------------------------
    StartFineTuning:
    ----------------------------------------------------------------------------
    Unfreezes the last N blocks of the pretrained encoder for fine-tuning.
    This is useful when adapting the model to a new dataset after pretraining.

    Parameters:
        blocks_to_unfreeze (int): Number of final encoder blocks to make trainable.
    """
    def StartFineTuning(self, blocks_to_unfreeze=1):
        total = len(self.pretrainedmodel.blocks)

        # Iterate over last 'blocks_to_unfreeze' blocks and enable gradients
        for idx in range(total - blocks_to_unfreeze, total):
            for p in self.pretrainedmodel.blocks[idx].parameters():
                p.requires_grad = True
