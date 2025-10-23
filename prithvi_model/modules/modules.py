import torch
import torch.nn as nn
import torch.nn.functional as F

"""
===============================================================================
DoubleConv: Fundamental convolutional building block
===============================================================================

Performs two consecutive 3×3 convolutional operations with BatchNorm and ReLU.
This block is a standard design used in encoder-decoder architectures (e.g., UNet),
providing nonlinear feature transformation while preserving spatial resolution.

Mathematical Form:
Let X ∈ ℝ^{C_in × H × W}
    → Conv1(3×3) → BN → ReLU → Conv2(3×3) → BN → ReLU
Output Y ∈ ℝ^{C_out × H × W}

Both convolutions use `padding=1`, hence:
    Output Height = Input Height
    Output Width  = Input Width
"""


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channels=None, bias=False):
        super().__init__()

        # If mid_channels is not provided, use the same as output
        if not mid_channels:
            mid_channels = out_channel

        # Sequentially apply two convolution layers with normalization and ReLU activation
        self.doubleconv = nn.Sequential(
            # First convolutional layer: input → mid-level features
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,  # maintain H, W dimensions
                bias=bias
            ),
            nn.BatchNorm2d(mid_channels),  # normalize activations
            nn.ReLU(inplace=True),  # nonlinearity

            # Second convolutional layer: mid-level → output features
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,  # maintain H, W
                bias=bias
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Input shape: (B, C_in, H, W)
        # Output shape: (B, C_out, H, W)
        return self.doubleconv(x)


"""
===============================================================================
outConv: 1×1 Convolution Layer
===============================================================================

Performs a pointwise convolution (1×1 kernel), often used to:
- Adjust the number of channels without changing spatial resolution.
- Project high-dimensional feature maps to class logits or reduced embeddings.

For input tensor X ∈ ℝ^{C_in × H × W}:
    Y = W * X + b
where W ∈ ℝ^{C_out × C_in × 1 × 1}
"""


class outConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 1×1 convolution for channel projection
        self.outconv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, image):
        # Direct linear mapping across channel dimension
        return self.outconv(image)


"""
===============================================================================
AdaptivePooling: Adaptive Average Pooling
===============================================================================

This module pools input feature maps into a fixed spatial scale regardless of
input size. Used in pyramid pooling to capture global context at multiple scales.

Given input X ∈ ℝ^{C × H × W}, output is Y ∈ ℝ^{C × s × s}
where `s` is defined by the `scale` parameter.
"""


class AdaptivePooling(nn.Module):
    def __init__(self, scale):
        super().__init__()
        # Create adaptive average pooling layer to downsample to (scale × scale)
        self.adapool = nn.AdaptiveAvgPool2d(scale)

    def forward(self, feature_map):
        # Output has fixed spatial resolution = scale
        return self.adapool(feature_map)


"""
===============================================================================
PyramidPooling: Multi-scale context aggregation module
===============================================================================

Captures features at multiple spatial scales using adaptive pooling operations.
Each pooled feature map represents a different receptive field, and all are
upsampled and concatenated back to the original resolution.

The module effectively builds a feature pyramid of global and local context.

Given:
    feature_map ∈ ℝ^{B × C_in × H × W}

Operations:
    - Adaptive pooling to scales [1, 2, 3, 6]
    - 1×1 convolution (reduce channels)
    - Upsample to original (H, W)
    - Concatenate all pooled features + original map → ℝ^{B × 5*C_in × H × W}
    - Fuse via DoubleConv to produce output ℝ^{B × C_out × H × W}
"""


class PyramidPooling(nn.Module):
    def __init__(self, scales=[1, 2, 3, 6], in_channel=768, out_channel=256, mid_channel=192):
        super().__init__()
        self.scales = scales  # multiple spatial context scales

        # Define adaptive pooling layers dynamically for each scale
        for i in range(1, 5):
            setattr(
                self, f"adapool{i}",
                AdaptivePooling(
                    scale=scales[i - 1]
                )
            )

        # 1×1 conv to reduce channel dimension before upsampling
        self.conv = outConv(in_channel=in_channel, out_channel=mid_channel)
        self.bn = nn.BatchNorm2d(mid_channel)

        # Final fusion convolution: merges concatenated feature maps
        self.ppm_final_conv = DoubleConv(
            in_channel=2 * in_channel,  # concatenated feature channels
            out_channel=out_channel,
            mid_channels=in_channel
        )

    def forward(self, feature_map):
        # feature_map: (B, C, H, W)
        _, _, H, W = feature_map.shape

        # Upsampling layer to bring pooled maps back to original size
        upsample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)

        # Scale 1 pooling: global context (entire image compressed to 1×1)
        pp1 = self.adapool1(feature_map)
        pp1 = self.conv(pp1)
        pp1 = self.bn(pp1)
        pp1 = F.relu(pp1)
        pp1 = upsample(pp1)  # expand back to (H, W)

        # Scale 2 pooling: coarse mid-scale context
        pp2 = self.adapool2(feature_map)
        pp2 = self.conv(pp2)
        pp2 = self.bn(pp2)
        pp2 = F.relu(pp2)
        pp2 = upsample(pp2)

        # Scale 3 pooling: medium context
        pp3 = self.adapool3(feature_map)
        pp3 = self.conv(pp3)
        pp3 = self.bn(pp3)
        pp3 = F.relu(pp3)
        pp3 = upsample(pp3)

        # Scale 6 pooling: fine-grained local context
        pp4 = self.adapool4(feature_map)
        pp4 = self.conv(pp4)
        pp4 = self.bn(pp4)
        pp4 = F.relu(pp4)
        pp4 = upsample(pp4)

        # Concatenate all pooled representations with the original feature map
        # Resulting channel dimension: C_total = C_in * (1 + number_of_scales)
        output = torch.cat((feature_map, pp1, pp2, pp3, pp4), dim=1)

        # Fuse concatenated feature maps using DoubleConv for spatial mixing
        output = self.ppm_final_conv(output)

        # Output shape: (B, out_channel, H, W)
        return output


"""
===============================================================================
FeaturePyramidNetwork (FPN): Multi-scale feature refinement
===============================================================================

Implements a top-down feature fusion mechanism similar to FPN.

Given two feature maps of different scales:
    - feature_map: higher-level, semantically rich (low resolution)
    - previous_feature_map: lower-level, detailed (higher resolution)

The module:
1. Projects feature_map via 1×1 conv.
2. Upsamples to match resolution of previous_feature_map.
3. Adds both (feature fusion).
4. Applies DoubleConv smoothing.

Mathematically:
    Y = Smooth( Upsample(Conv(feature_map)) + Upsample(previous_feature_map) )

Output dimension:
    ℝ^{B × out_channel × (H×scale) × (W×scale)}
"""


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channel=768, out_channel=256):
        super().__init__()
        # 1×1 conv for channel reduction of high-level feature
        self.conv = outConv(in_channel=in_channel, out_channel=out_channel)

        # Smoothing convolution to refine fused features
        self.smoothingconv = DoubleConv(in_channel=out_channel, out_channel=out_channel)

    def forward(self, feature_map, previous_feature_map, scale):
        # feature_map: high-level features
        # previous_feature_map: from shallower layer

        # Project high-level features to a lower channel space
        feature_map = self.conv(feature_map)
        feature_map = F.relu(feature_map)

        # Upsample to match the spatial resolution of lower feature map
        feature_map = F.interpolate(input=feature_map, scale_factor=scale, mode='bilinear', align_corners=True)

        # Upsample previous feature map (ensures same dimension)
        previous_feature_map = F.interpolate(input=previous_feature_map, scale_factor=2, mode='bilinear',
                                             align_corners=True)

        # Element-wise addition (feature fusion)
        output = feature_map + previous_feature_map

        # Smooth combined feature to remove checkerboard artifacts
        output = self.smoothingconv(output)

        # Output: refined multi-scale feature representation
        return output
