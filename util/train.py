# -*- coding: utf-8 -*-
"""
Training + validation loop for a U-PerNet-style Prithvi.

- Framework: PyTorch
- Optimizer: Adam
- Loss: Linear combination of DiceLoss + FocalLoss (Kornia) with ignore_index for masked pixels
- Metric: torchmetrics MulticlassAccuracy (also ignoring the same index)
- Progress: tqdm progress bars
- Device: CPU or CUDA
- Dataloaders:
    * Train loader shuffles; Validation loader does not shuffle
- Returns:
    * prithvi_model (with updated weights)
    * train_loss_list (per-epoch average training loss)
    * train_accuracies_list (per-epoch training accuracy)
    * val_loss_list (per-epoch average validation loss)
    * val_accuracies_list (per-epoch validation accuracy)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from kornia.losses import DiceLoss, FocalLoss
from typing import Optional, Tuple
import os, pickle


def save_epoch_model(model, save_dir, epoch):
    """
    Save prithvi_model state dict every 3rd epoch with epoch suffix.

    This function saves the prithvi_model's state dictionary with an epoch identifier
    to allow tracking of prithvi_model progress throughout training. Saves only
    every 3rd epoch to conserve disk space.

    Args:
        model (torch.nn.Module): The PyTorch prithvi_model to save
        save_dir (str): Directory path where the prithvi_model checkpoint will be saved
        epoch (int): Current epoch number (0-indexed)

    Returns:
        None

    Side Effects:
        - Creates a .pt file in save_dir with format {model_base_name}_ep{epoch+1}.pt
        - Prints confirmation message of successful save

    Example:
        For epoch=2 (3rd epoch):
        Creates file: 'Prithvi_300M_ep3.pt'
    """

    # Only save every 3rd epoch to save disk space
    if (epoch + 1) % 3 != 0:
        return

    # create dir if path doesnt exist
    os.makedirs(save_dir, exist_ok=True)

    epoch_model_file = f"Prithvi_300M_ep{epoch + 1}.pt"

    # Save prithvi_model state dict with epoch suffix
    torch.save(model.state_dict(), os.path.join(save_dir, epoch_model_file))
    print(f"Saved epoch {epoch + 1} prithvi_model: {epoch_model_file}")


def save_final_model_and_metrics(model, save_dir, train_metrics_per_city, val_metrics_per_city):
    """
    Save final prithvi_model, configuration file, and all training metrics at the end of training.

    This function performs the final save operation after all epochs are complete.
    It saves three types of files:
    1. Model state dictionary (.pt file)
    2. Model configuration as JSON (.json file)
    3. Training and validation metrics as pickle files (.pkl files)

    Args:
        model (torch.nn.Module): The trained PyTorch prithvi_model with config_file attribute
        model_name (str): Name of the prithvi_model architecture. Must be one of:
                         'Base UNet', 'Attention Block UNet', 'Hybrid UNet', 'Self Attention UNet'
        save_dir (str): Directory path where all files will be saved
        train_metrics_per_city (list): List of tuples containing (train_loss, train_acc)
                                      for each city/tile training session
        val_metrics_per_city (list): List of tuples containing (val_loss, val_acc)
                                    for each city/tile validation session

    Returns:
        None

    Side Effects:
        - Creates prithvi_model .pt file with final trained weights
        - Creates config .json file with prithvi_model configuration parameters
        - Creates train metrics .pkl file with training loss/accuracy history
        - Creates validation metrics .pkl file with validation loss/accuracy history
        - Prints confirmation messages for all saved files

    Raises:
        AttributeError: If prithvi_model does not have config_file attribute
        OSError: If save_dir is not writable or does not exist

    Example Files Created (for 'Base UNet'):
        - base_unet.pt (prithvi_model weights)
        - base_unet_config.json (prithvi_model configuration)
        - train_metrics_unet_model_BASE.pkl (training metrics)
        - val_metrics_model_BASE.pkl (validation metrics)
    """
    # create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Mapping of prithvi_model names to their respective file naming conventions
    # Format: (model_file, train_metrics_file, val_metrics_file)
    model_file, train_file, val_file = 'Pritvi_300M_.pt', 'train_metrics_prithvi.pkl', 'val_metrics_prithvi.pkl'


    # Save final prithvi_model state dictionary containing learned parameters
    torch.save(model.state_dict(), os.path.join(save_dir, model_file))

    # Save training metrics (loss and accuracy per city/tile)
    with open(os.path.join(save_dir, train_file), 'wb') as f:
        pickle.dump(train_metrics_per_city, f)

    # Save validation metrics (loss and accuracy per city/tile)
    with open(os.path.join(save_dir, val_file), 'wb') as f:
        pickle.dump(val_metrics_per_city, f)

    # Provide user feedback on successful saves
    print(f"Saved final prithvi_model: {model_file}")
    print(f"Saved metrics: {train_file}, {val_file}")


def train_one_epoch(
        model: nn.Module,
        train_ds,
        val_ds,
        num_classes: int,
        device: Optional[torch.device | str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 7e-5,
        ignore_index: int = 255,
        # loss config
        dice_w: float = 0.5,
        focal_w: float = 1.0,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,  # optional per-class weights
) -> Tuple[nn.Module, list[float], list[float], list[float], list[float]]:
    """
    Train and validate a segmentation prithvi_model on the given datasets.

    Args:
        model (nn.Module): The segmentation network that outputs logits of shape [B, C, H, W].
        train_ds (Dataset): Training dataset; each item returns (image, label) where:
                            image -> FloatTensor [C, H, W], label -> LongTensor [H, W].
        val_ds (Dataset): Validation dataset; same format as train_ds.
        num_classes (int): Number of classes for segmentation (used for accuracy).
        device (str or torch.device): Device to use for training ('cpu', 'cuda', 'xpu', etc.).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for both train and validation loaders.
        lr (float): Learning rate for Adam optimizer.
        ignore_index (int): Label value to ignore in loss/metrics (e.g., 255 for unlabeled).
        dice_w (float): Weight for Dice loss component in the final loss.
        focal_w (float): Weight for Focal loss component in the final loss.
        focal_gamma (float): Gamma parameter for Focal loss.
        class_weights (torch.Tensor, optional): Optional per-class weights. If provided,
                                                passed to DiceLoss(weight=...) and FocalLoss(alpha=...).

    Returns:
        Tuple:
            - prithvi_model (nn.Module): The trained prithvi_model instance.
            - train_loss_list (List[float]): Per-epoch average training loss.
            - train_accuracies_list (List[float]): Per-epoch training accuracy.
            - val_loss_list (List[float]): Per-epoch average validation loss.
            - val_accuracies_list (List[float]): Per-epoch validation accuracy.
    """

    # ---------------------------- DEVICE -------------------------------------
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Move model to device
    # NOTE: DataParallel disabled due to nested pretrained model architecture issues
    # The terratorch pretrained backbone doesn't play well with DataParallel
    model = model.to(device=device)
    print(f"Using device: {device}")

    # --------------------------- DATALOADERS ---------------------------------
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

    # --------------------------- LOSS FUNCTIONS ------------------------------
    # DiceLoss: penalizes overlap mismatch (from_logits=True since prithvi_model outputs logits)
    # Make sure class_weights is moved to correct device/dtype before usage
    cw = class_weights.to(device=device, dtype=torch.float32) if class_weights is not None else None

    dice_criterion = DiceLoss(
        average='micro',
        ignore_index=ignore_index,
        weight=cw,
    )

    focal_criterion = FocalLoss(
        alpha=None,  # alpha should be float or None
        gamma=focal_gamma,
        weight=cw,  # safe tensor or None
        reduction="mean",
        ignore_index=ignore_index,
    )

    # --------------------------- OPTIMIZER -----------------------------------
    opt = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # Optimize with Intel XPU if available

    # --------------------------- METRICS -------------------------------------
    # All metrics on CPU for consistency
    accuracy = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index).to("cpu")
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average=None).to("cpu")
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index, average='macro').to("cpu")
    precision_metric = MulticlassPrecision(num_classes=num_classes, ignore_index=ignore_index, average=None).to("cpu")
    recall_metric = MulticlassRecall(num_classes=num_classes, ignore_index=ignore_index, average=None).to("cpu")
    f1_metric = MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average=None).to("cpu")  # Per-class Dice
    mean_f1_metric = MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average='macro').to("cpu")  # Mean Dice
    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index, normalize='true').to("cpu")

    # --------------------------- LOGGING -------------------------------------
    train_loss_list, val_loss_list = [], []
    train_accuracies_list, val_accuracies_list = [], []

    # ========================== EPOCH LOOP ===================================
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader)
        
        # Per-class pixel counters for training
        train_class_counts = torch.zeros(num_classes, dtype=torch.long)

        # ============================ TRAIN ==================================
        for images, labels in train_loader_tqdm:
            images = images.to(device=device)
            labels = labels.to(device=device)

            print(images.shape)
            print(labels.shape)

            opt.zero_grad()  # Reset gradients
            logits = model(images)  # Forward pass
            # ----- Compute Linear Combination Loss -----
            dice_loss = dice_criterion(logits, labels)
            focal_loss = focal_criterion(logits, labels)
            loss = dice_w * dice_loss + focal_w * focal_loss

            # Backpropagation
            loss.backward()
            opt.step()

            # Track running loss
            running_loss += loss.item() / len(train_loader)

            # Update accuracy (using predictions on CPU)
            preds = logits.argmax(dim=1).detach().cpu()
            labels_cpu = labels.detach().cpu()
            accuracy.update(preds, labels_cpu)
            
            # Count pixels per class for support
            for c in range(num_classes):
                train_class_counts[c] += ((labels_cpu == c) & (labels_cpu != ignore_index)).sum()

            train_loader_tqdm.set_postfix({
                "Training Loss": f"{running_loss:.4f}",
                "Training Acc": f"{accuracy.compute().item():.4f}"
            })

        # Store epoch-level training stats
        train_accuracy = accuracy.compute().item()
        train_accuracies_list.append({
            'accuracy': train_accuracy,
            'class_support': train_class_counts.tolist()
        })
        train_loss_list.append(running_loss)
        accuracy.reset()

        # =========================== VALIDATE ================================
        model.eval()
        val_running = 0.0
        focal_loss_run, dice_loss_run = 0.0, 0.0
        
        # Per-class pixel counters for validation
        val_class_counts = torch.zeros(num_classes, dtype=torch.long)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device=device)
                labels = labels.to(device=device)
                logits = model(images)

                # ----- Compute Linear Combination Loss -----
                dice_loss = dice_criterion(logits, labels)
                focal_loss = focal_criterion(logits, labels)
                loss = dice_w * dice_loss + focal_w * focal_loss

                val_running += loss.item() / len(val_loader)
                dice_loss_run += dice_loss.item() / len(val_loader)
                focal_loss_run += focal_loss.item() / len(val_loader)

                preds = logits.argmax(dim=1).detach().cpu()
                labels_cpu = labels.detach().cpu()
                
                # Update all metrics
                accuracy.update(preds, labels_cpu)
                iou_metric.update(preds, labels_cpu)
                miou_metric.update(preds, labels_cpu)
                precision_metric.update(preds, labels_cpu)
                recall_metric.update(preds, labels_cpu)
                f1_metric.update(preds, labels_cpu)
                mean_f1_metric.update(preds, labels_cpu)
                confusion_matrix.update(preds, labels_cpu)
                
                # Count pixels per class for support
                for c in range(num_classes):
                    val_class_counts[c] += ((labels_cpu == c) & (labels_cpu != ignore_index)).sum()

        # Compute all validation metrics
        val_accuracy = accuracy.compute().item()
        per_class_iou = iou_metric.compute().tolist()
        mean_iou = miou_metric.compute().item()
        per_class_precision = precision_metric.compute().tolist()
        per_class_recall = recall_metric.compute().tolist()
        per_class_dice = f1_metric.compute().tolist()  # F1 = Dice
        overall_dice = mean_f1_metric.compute().item()  # Mean F1 = Mean Dice
        conf_matrix = confusion_matrix.compute().tolist()
        
        # Compute mean class accuracy from per-class recall
        mean_class_acc = torch.tensor(per_class_recall).mean().item()
        
        # Store comprehensive validation metrics
        val_loss_list.append({
            'total_loss': val_running,
            'dice_loss': dice_loss_run,
            'focal_loss': focal_loss_run
        })
        
        val_accuracies_list.append({
            'overall_accuracy': val_accuracy,
            'mean_iou': mean_iou,
            'per_class_iou': per_class_iou,
            'mean_class_accuracy': mean_class_acc,
            'overall_dice': overall_dice,
            'per_class_dice': per_class_dice,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'confusion_matrix': conf_matrix,
            'class_support': val_class_counts.tolist()
        })
        
        # Reset all metrics
        accuracy.reset()
        iou_metric.reset()
        miou_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        mean_f1_metric.reset()
        confusion_matrix.reset()

        # ---------------------------- LOGGING --------------------------------
        print(f'''epoch [{epoch + 1}/{epochs}]
        \t training loss: {running_loss:.4f},
        \t validation loss: {val_running:.4f},
        \t Validation Dice Loss: {dice_loss_run:.4f},
        \t Validation focal loss: {focal_loss_run:.4f}
        \t Train Accuracy: {train_accuracy:.4f},
        \t Val Accuracy: {val_accuracy:.4f}
        \t Val mIoU: {mean_iou:.4f}
        \t Val mAcc: {mean_class_acc:.4f}
        \t Val Overall Dice: {overall_dice:.4f}
        ''')

    # --------------------------- RETURN --------------------------------------
    return model, train_loss_list, train_accuracies_list, val_loss_list, val_accuracies_list
