
import torch

import os

from util import train_one_epoch, save_epoch_model, save_final_model_and_metrics
from dataset_generator import HLSDataset


os.environ['TORCH_DYNAMO_DISABLE_DOCSTRING_CHECKS'] = '1'

from torchsummary import summary
import numpy as np
from terratorch.registry import BACKBONE_REGISTRY

from prithvi_model import Prithvi_EO

def train_model(model, image_paths, label_path, device, batch_size,
                epochs, save_dir, train_params, dataset_params):
    print("Training...")

    train_metrics_per_city = []
    val_metrics_per_city = []



    # summary of the prithvi_model
    print(summary(model, input_size = (6,1,224,224)))


    # Main training loop: iterate through specified number of epochs
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        shuffled_indexes = np.arange(len(image_paths))
        np.random.shuffle(shuffled_indexes)

        # Train on each city in the randomized order
        for idx in shuffled_indexes:
            image_path = image_paths[idx]

            # Create dataset for current city with user-specified parameters
            # Sentinel2Dataset should handle loading and preprocessing of satellite imagery
            dataset = HLSDataset(
                image_path,
                label_path,
                tile=dataset_params['tile'],  # Tile size for cropping
                stride=dataset_params['stride'],  # Sliding window stride
                ignore_index=dataset_params['ignore_index'],  # Ignore value in labels
                verbose= True,
            )

            # Split dataset into training and validation sets (80/20 split)
            TRAIN_SPLIT = int(len(dataset) * 0.8)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset,
                [TRAIN_SPLIT, len(dataset) - TRAIN_SPLIT]
            )

            print(f"Currently starting tile {os.path.basename(label_path)}")

            # Train prithvi_model on current city's data for one epoch
            # train_unet should be defined elsewhere and handle the actual training loop
            model, train_loss, train_acc, val_loss, val_acc = train_one_epoch(
                model=model,  # Model to train
                train_ds=train_ds,  # Training dataset
                val_ds=val_ds,  # Validation dataset
                num_classes=4,  # Number of output classes
                device=device,  # Computation device
                epochs=1,  # Single epoch per city
                batch_size=batch_size,  # Batch size for training
                lr=train_params['lr'],  # Learning rate
                ignore_index=train_params['ignore_index'],  # Index to ignore in loss
                dice_w=train_params['dice_w'],  # Dice loss weight
                focal_w=train_params['focal_w'],  # Focal loss weight
                focal_gamma=train_params['focal_gamma'],  # Focal loss gamma
                class_weights=train_params['class_weights']  # Class balancing weights
            )

            # Store metrics for this city's training session
            train_metrics_per_city.append((train_loss, train_acc))
            val_metrics_per_city.append((val_loss, val_acc))

        # Save prithvi_model checkpoint after completing all cities for this epoch
        save_epoch_model(model, save_dir, epoch)

    # Save final prithvi_model, configuration, and all collected metrics
    save_final_model_and_metrics(
        model, save_dir,
        train_metrics_per_city, val_metrics_per_city
    )




# ==== Required Inputs ====
image_paths = ["./Dataset/HLS-2/Orlando/HLS.S30.T17RMM.2024098T155819.v2.0.B02.tif", # orlando
               "./Dataset/HLS-2/Seattle/HLS.S30.T10TET.2025159T190909.v2.0.B02.tif", # seattle
               "./Dataset/HLS-2/Los Angeles/HLS.S30.T11SLT.2024128T182921.v2.0.B02.tif", # Los Angeles
               "./Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif", # chicago
               ]
label_paths = "./Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"  # United states of america

# ==== Training parameters ====
batch_size = 32  # Default: 32
device = 'auto'  # Options: 'auto', 'cuda', 'cpu', 'xpu'
epochs = 10  # Default: 10
lr = 1e-4  # Learning rate
ignore_index = 255  # Label value to ignore in loss
dice_weight = 0.5  # Dice loss weight
focal_weight = 1.0  # Focal loss weight
focal_gamma = 2.0  # Focal loss gamma
class_weights = None  # TODO: Add class weights if needed, e.g., [1.0, 2.0, 1.5, 0.8]

# ==== Dataset parameters ====
tile_size = 224  # Crop size
stride = 224  # Sliding window step

# ==== Output settings ====
save_dir = 'prithvi_model/saved_models'  # Default output dir

# ==== Example usage in code ====
print("Training config:")
print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
print(f"Device: {device}")
print(f"Save Directory: {save_dir}")

# Setup device
if device == 'auto':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('mps') if torch.backends.mps.is_available() else device
    device = torch.device('xpu') if torch.xpu.is_available() else device
else:
    device = torch.device(device)
print(f"Using device: {device}")

# Create save directory
os.makedirs(save_dir, exist_ok=True)

# Prepare training parameters
train_params = {
    'lr':lr,
    'ignore_index':ignore_index,
    'dice_w':dice_weight,
    'focal_w':focal_weight,
    'focal_gamma':focal_gamma,
    'class_weights':class_weights
}

# Prepare dataset parameters
dataset_params = {
    'tile': tile_size,
    'stride':stride,
    'ignore_index': ignore_index,
}


pretrained_model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True) # load prithvi pretrained prithvi_model

model = Prithvi_EO(
    pretrained_model=pretrained_model,
    num_classes=4,
    fpn_blocks=[6, 12, 18, 24],
    scale_factors=[1, 2, 3, 6],
    embed_dim=1024,
    out_channels_feature_map=256,
    FPN_out_channels=256,
    upsampling_scale_list=[4, 2, 1, 0.5],
    u_height = 56,
    u_width = 56,
)




train_model(model, image_paths, label_paths, device, batch_size, epochs, save_dir, train_params, dataset_params)

print("Training completed!")