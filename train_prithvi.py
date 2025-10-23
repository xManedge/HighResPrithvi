import torch
import os

from util import train_one_epoch, save_epoch_model, save_final_model_and_metrics
from dataset_generator import HLSDataset

os.environ['TORCH_DYNAMO_DISABLE_DOCSTRING_CHECKS'] = '1'

from torchsummary import summary
import numpy as np
from terratorch.registry import BACKBONE_REGISTRY
from prithvi_model import Prithvi_EO

"""
================================================================================
                            MODEL TRAINING SCRIPT
================================================================================
This script performs fine-tuning or training of the Prithvi_EO segmentation model
on HLS satellite imagery.

It handles:
    - Dataset loading and tiling from large HLS scenes
    - Model setup and summary display
    - Training and validation across multiple epochs
    - Per-city training loop for multiple regions
    - Checkpoint saving after each epoch
    - Final model saving and metric collection

Mathematical / Conceptual Notes:
--------------------------------
Each input image tile (size 224×224 pixels × spectral bands) is processed through
the model in batches. The model output is a multi-class segmentation map of shape:
    (B, num_classes, H, W)
where H = W = 224 by default.

Loss is computed per-pixel using Dice + Focal combinations, with an ignore_index
to mask invalid regions.
"""


def train_model(model, image_paths, label_path, device, batch_size,
                epochs, save_dir, train_params, dataset_params):
    """Main function that manages the entire training and validation process."""

    print("Training...")

    # These lists will store per-city performance metrics for later evaluation.
    train_metrics_per_city = []
    val_metrics_per_city = []

    """
    ---------------------------------------------------------------------------
    MODEL SUMMARY
    ---------------------------------------------------------------------------
    Display the structure and parameter count of the Prithvi_EO model.
    The input size is (6, 1, 224, 224):
        - 6: number of spectral bands (HLS-like)
        - 1: temporal dimension (single time slice)
        - 224x224: spatial tile size
    """
    print(summary(model, input_size=(6, 1, 224, 224)))

    """
    ===========================================================================
    EPOCH LOOP
    ===========================================================================
    The model trains for the specified number of epochs.
    Each epoch will iterate over all provided image paths (i.e., cities/regions).
    """
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        # Randomly shuffle city order to reduce bias across epochs.
        shuffled_indexes = np.arange(len(image_paths))
        np.random.shuffle(shuffled_indexes)

        """
        -----------------------------------------------------------------------
        CITY LOOP
        -----------------------------------------------------------------------
        Each city corresponds to one HLS granule or region image path.
        For each city:
            - The HLS dataset is created
            - It is split into training and validation sets (80/20)
            - The model trains for one epoch on this city's tiles
        """
        for idx in shuffled_indexes:
            image_path = image_paths[idx]

            # Dataset creation for a single HLS scene
            # HLSDataset handles raster reading, tiling, and preprocessing.
            dataset = HLSDataset(
                image_path,
                label_path,
                tile=dataset_params['tile'],  # (224,224) tile extraction
                stride=dataset_params['stride'],  # controls tile overlap
                ignore_index=dataset_params['ignore_index'],
                verbose=True,
            )

            """
            ---------------------------------------------------------------
            TRAIN/VALIDATION SPLIT
            ---------------------------------------------------------------
            80% of tiles are used for training, 20% for validation.
            """
            TRAIN_SPLIT = int(len(dataset) * 0.8)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset,
                [TRAIN_SPLIT, len(dataset) - TRAIN_SPLIT]
            )

            print(f"Currently starting tile {os.path.basename(label_path)}")

            """
            ---------------------------------------------------------------
            TRAINING FOR CURRENT CITY
            ---------------------------------------------------------------
            The function `train_one_epoch` handles:
                - Dataloader creation
                - Forward/backward passes
                - Loss computation (Dice + Focal)
                - Metric computation (accuracy, IoU, etc.)

            Output:
                model, train_loss, train_acc, val_loss, val_acc
            """
            model, train_loss, train_acc, val_loss, val_acc = train_one_epoch(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                num_classes=4,
                device=device,
                epochs=1,  # One mini-epoch per city
                batch_size=batch_size,
                lr=train_params['lr'],
                ignore_index=train_params['ignore_index'],
                dice_w=train_params['dice_w'],
                focal_w=train_params['focal_w'],
                focal_gamma=train_params['focal_gamma'],
                class_weights=train_params['class_weights']
            )

            # Save performance metrics for later averaging or plotting
            train_metrics_per_city.append((train_loss, train_acc))
            val_metrics_per_city.append((val_loss, val_acc))

        """
        -----------------------------------------------------------------------
        CHECKPOINT SAVING
        -----------------------------------------------------------------------
        After each epoch (i.e., after training across all cities),
        save the model checkpoint for recovery or analysis.
        """
        save_epoch_model(model, save_dir, epoch)

    """
    ===========================================================================
    FINAL MODEL SAVING
    ===========================================================================
    Once all epochs are completed, save the final trained weights
    along with all recorded metrics for reproducibility and later use.
    """
    save_final_model_and_metrics(
        model, save_dir,
        train_metrics_per_city, val_metrics_per_city
    )


"""
==============================================================================
CONFIGURATION AND INITIALIZATION SECTION
==============================================================================
Below: all hyperparameters, dataset paths, and training settings are defined.
These control the entire training pipeline behavior.
"""

# ==== Required Inputs ====
image_paths = [
    "./Dataset/HLS-2/Orlando/HLS.S30.T17RMM.2024098T155819.v2.0.B02.tif",
    "./Dataset/HLS-2/Seattle/HLS.S30.T10TET.2025159T190909.v2.0.B02.tif",
    "./Dataset/HLS-2/Los Angeles/HLS.S30.T11SLT.2024128T182921.v2.0.B02.tif",
    "./Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif",
]
label_paths = "./Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"

# ==== Training parameters ====
batch_size = 32
device = 'auto'
epochs = 10
lr = 1e-4
ignore_index = 255
dice_weight = 0.5
focal_weight = 1.0
focal_gamma = 2.0
class_weights = None

# ==== Dataset parameters ====
tile_size = 224
stride = 224

# ==== Output settings ====
save_dir = 'prithvi_model/saved_models'

"""
---------------------------------------------------------------------------
DISPLAY TRAINING CONFIGURATION
---------------------------------------------------------------------------
This helps confirm setup before training starts.
"""
print("Training config:")
print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
print(f"Device: {device}")
print(f"Save Directory: {save_dir}")

"""
==============================================================================
DEVICE SETUP
==============================================================================
Automatically select the best available device:
    - CUDA GPU
    - Apple MPS
    - Intel XPU
    - Otherwise CPU fallback
"""
if device == 'auto':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    device = torch.device('xpu') if torch.xpu.is_available() else device
else:
    device = torch.device(device)
print(f"Using device: {device}")

"""
==============================================================================
DIRECTORY PREPARATION
==============================================================================
Create the model save directory if it does not exist.
"""
os.makedirs(save_dir, exist_ok=True)

"""
==============================================================================
PARAMETER DICTIONARIES
==============================================================================
Define training and dataset parameters for convenience and readability.
These dictionaries are passed directly to helper functions.
"""
train_params = {
    'lr': lr,
    'ignore_index': ignore_index,
    'dice_w': dice_weight,
    'focal_w': focal_weight,
    'focal_gamma': focal_gamma,
    'class_weights': class_weights
}

dataset_params = {
    'tile': tile_size,
    'stride': stride,
    'ignore_index': ignore_index,
}

"""
==============================================================================
MODEL INITIALIZATION
==============================================================================
Load the pretrained backbone from the Terratorch registry.
Different versions correspond to different model sizes and token lengths.
    - 'prithvi_eo_v2_100_tl': 100M parameters (small ViT)
    - 'prithvi_eo_v2_300_tl': 300M parameters (large ViT)

The pretrained model serves as a frozen encoder backbone.
"""
# pretrained_model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True)  # 300M ViT-Large
pretrained_model = BACKBONE_REGISTRY.build("prithvi_eo_v2_100_tl", pretrained=True)  # 100M ViT-Small

"""
---------------------------------------------------------------------------
CREATE SEGMENTATION MODEL WRAPPER
---------------------------------------------------------------------------
The Prithvi_EO wrapper adds:
    - Pyramid pooling for multiscale context
    - FPN layers for hierarchical fusion
    - Upsampling and segmentation head for dense predictions
"""
model = Prithvi_EO(
    pretrained_model=pretrained_model,
    num_classes=4,
    fpn_blocks=[3, 6, 9, 12],  # Transformer layers used for skip features
    scale_factors=[1, 2, 3, 6],  # Pooling scales
    embed_dim=768,  # Embedding size (ViT-Small)
    out_channels_feature_map=128,
    FPN_out_channels=128,
    upsampling_scale_list=[4, 2, 1, 0.5],
    u_height=56,
    u_width=56,
)

"""
==============================================================================
START TRAINING
==============================================================================
"""
train_model(
    model, image_paths, label_paths,
    device, batch_size, epochs,
    save_dir, train_params, dataset_params
)

print("Training completed!")
