"""
Prithvi-EO UperNet Implementation

A UperNet-style decoder for semantic segmentation with the Prithvi Earth Observation encoder.
Implements Pyramid Pooling Module (PPM) and Feature Pyramid Network (FPN).

Usage:
    from prithvi_upernet import Prithvi_EO

    model = Prithvi_EO(
        pretrained_model=encoder,
        num_classes=4,
        embed_dim=768,
        out_channels_feature_map=256
    )

    output = model(x)  # x: (B, 6, 1, 224, 224) -> output: (B, num_classes, 224, 224)
"""

from .modules import (
    DoubleConv,
    outConv,
    AdaptivePooling,
    PyramidPooling,
    FeaturePyramidNetwork,
    Prithvi_EO,
)

__all__ = [
    'DoubleConv',
    'outConv',
    'AdaptivePooling',
    'PyramidPooling',
    'FeaturePyramidNetwork',
    'Prithvi_EO',
]

__version__ = '0.1.0'