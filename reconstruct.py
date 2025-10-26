import torch
import numpy as np
from torch.utils.data import DataLoader
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from torchmetrics.classification import (
    MulticlassAccuracy, 
    MulticlassJaccardIndex, 
    MulticlassConfusionMatrix,
    MulticlassPrecision, 
    MulticlassRecall, 
    MulticlassF1Score
)
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TORCH_DYNAMO_DISABLE_DOCSTRING_CHECKS'] = '1'

from terratorch.registry import BACKBONE_REGISTRY
from prithvi_model import Prithvi_EO
from dataset_generator import HLSDataset


class SegmentationMetricsTracker:
    """
    Comprehensive metrics tracking for image segmentation using torchmetrics.
    Matches the validation metrics from training.
    """

    def __init__(self, n_classes=4, class_names=None, ignore_index=255, device='cpu'):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'Class_{i}' for i in range(n_classes)]
        self.device = device

        # Initialize all torchmetrics (same as in training)
        self.accuracy = MulticlassAccuracy(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average='micro'
        ).to(device)
        
        self.iou_metric = MulticlassJaccardIndex(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average=None
        ).to(device)
        
        self.miou_metric = MulticlassJaccardIndex(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average='macro'
        ).to(device)
        
        self.precision_metric = MulticlassPrecision(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average=None
        ).to(device)
        
        self.recall_metric = MulticlassRecall(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average=None
        ).to(device)
        
        self.f1_metric = MulticlassF1Score(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average=None
        ).to(device)
        
        self.mean_f1_metric = MulticlassF1Score(
            num_classes=n_classes, 
            ignore_index=ignore_index, 
            average='macro'
        ).to(device)
        
        self.confusion_matrix = MulticlassConfusionMatrix(
            num_classes=n_classes, 
            ignore_index=ignore_index
        ).to(device)

        # For Cohen's Kappa (sklearn)
        self.all_predictions = []
        self.all_labels = []
        
        # Per-class pixel counters for support
        self.class_counts = torch.zeros(n_classes, dtype=torch.long)
        
        # Patch-level tracking
        self.patch_level_metrics = []

    def update(self, predictions, labels, patch_info=None):
        """
        Update metrics with new batch of predictions and labels.

        Args:
            predictions: torch.Tensor of shape (B, H, W)
            labels: torch.Tensor of shape (B, H, W)
            patch_info: dict with patch metadata (optional)
        """
        # Convert to tensors if needed
        if not torch.is_tensor(predictions):
            predictions = torch.from_numpy(predictions)
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels)
        
        # Move to device
        preds = predictions.to(self.device)
        labels_dev = labels.to(self.device)

        # Update all torchmetrics
        self.accuracy.update(preds, labels_dev)
        self.iou_metric.update(preds, labels_dev)
        self.miou_metric.update(preds, labels_dev)
        self.precision_metric.update(preds, labels_dev)
        self.recall_metric.update(preds, labels_dev)
        self.f1_metric.update(preds, labels_dev)
        self.mean_f1_metric.update(preds, labels_dev)
        self.confusion_matrix.update(preds, labels_dev)

        # Count pixels per class for support
        labels_cpu = labels_dev.cpu()
        for c in range(self.n_classes):
            self.class_counts[c] += ((labels_cpu == c) & (labels_cpu != self.ignore_index)).sum()

        # Store for Cohen's Kappa
        preds_cpu = preds.cpu()
        valid_mask = labels_cpu != self.ignore_index
        self.all_predictions.extend(preds_cpu[valid_mask].flatten().tolist())
        self.all_labels.extend(labels_cpu[valid_mask].flatten().tolist())

        # Calculate patch-level metrics if needed
        if patch_info is not None:
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                patch_metrics = self._calculate_patch_metrics(
                    preds_cpu[i], 
                    labels_cpu[i]
                )
                patch_metrics.update(patch_info if i == 0 else {})
                self.patch_level_metrics.append(patch_metrics)

    def _calculate_patch_metrics(self, pred_patch, label_patch):
        """Calculate metrics for a single patch."""
        metrics = {}
        
        # Flatten and filter
        pred_flat = pred_patch.flatten()
        label_flat = label_patch.flatten()
        valid_mask = label_flat != self.ignore_index
        
        pred_valid = pred_flat[valid_mask]
        label_valid = label_flat[valid_mask]
        
        if len(pred_valid) > 0:
            metrics['patch_accuracy'] = (pred_valid == label_valid).float().mean().item()
            metrics['total_pixels'] = len(pred_valid)
            
            # Per-class pixel counts
            for class_id in range(self.n_classes):
                pred_count = (pred_valid == class_id).sum().item()
                label_count = (label_valid == class_id).sum().item()
                correct_count = ((pred_valid == class_id) & (label_valid == class_id)).sum().item()
                
                metrics[f'{self.class_names[class_id]}_pred_pixels'] = pred_count
                metrics[f'{self.class_names[class_id]}_true_pixels'] = label_count
                metrics[f'{self.class_names[class_id]}_correct_pixels'] = correct_count
                
                if pred_count > 0:
                    metrics[f'{self.class_names[class_id]}_precision'] = correct_count / pred_count
                else:
                    metrics[f'{self.class_names[class_id]}_precision'] = 0.0
                
                if label_count > 0:
                    metrics[f'{self.class_names[class_id]}_recall'] = correct_count / label_count
                else:
                    metrics[f'{self.class_names[class_id]}_recall'] = 1.0
        
        return metrics

    def compute_all_metrics(self):
        """
        Compute all metrics (matching training validation metrics).
        Returns a comprehensive dictionary.
        """
        metrics = {}

        # 1. Overall Accuracy
        metrics['accuracy'] = self.accuracy.compute().item()

        # 2. Per-class IoU
        per_class_iou = self.iou_metric.compute()
        metrics['per_class_iou'] = per_class_iou.cpu().tolist()
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_iou'] = per_class_iou[i].item()

        # 3. Mean IoU
        metrics['mean_iou'] = self.miou_metric.compute().item()

        # 4. Per-class Precision
        per_class_precision = self.precision_metric.compute()
        metrics['per_class_precision'] = per_class_precision.cpu().tolist()
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = per_class_precision[i].item()

        # 5. Per-class Recall
        per_class_recall = self.recall_metric.compute()
        metrics['per_class_recall'] = per_class_recall.cpu().tolist()
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_recall'] = per_class_recall[i].item()

        # 6. Per-class F1/Dice Score
        per_class_dice = self.f1_metric.compute()
        metrics['per_class_dice'] = per_class_dice.cpu().tolist()
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_dice'] = per_class_dice[i].item()
            metrics[f'{class_name}_f1'] = per_class_dice[i].item()

        # 7. Overall Dice/Mean F1
        metrics['overall_dice'] = self.mean_f1_metric.compute().item()
        metrics['mean_f1'] = metrics['overall_dice']

        # 8. Confusion Matrix
        conf_matrix = self.confusion_matrix.compute()
        metrics['confusion_matrix'] = conf_matrix.cpu().tolist()

        # 9. Cohen's Kappa Score (using sklearn)
        if len(self.all_predictions) > 0:
            metrics['cohen_kappa'] = cohen_kappa_score(
                self.all_labels, 
                self.all_predictions
            )
        else:
            metrics['cohen_kappa'] = 0.0

        # 10. Per-class support
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_support'] = self.class_counts[i].item()

        # 11. Macro/Micro averages
        metrics['macro_precision'] = per_class_precision.mean().item()
        metrics['macro_recall'] = per_class_recall.mean().item()
        metrics['macro_f1'] = per_class_dice.mean().item()

        return metrics

    def get_confusion_matrix(self):
        """Get confusion matrix as numpy array."""
        return self.confusion_matrix.compute().cpu().numpy()

    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """Plot and optionally save confusion matrix."""
        cm = self.get_confusion_matrix()

        plt.figure(figsize=(10, 8))

        if normalize:
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Normalized Confusion Matrix')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix (Pixel Counts)')

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")

        return plt.gcf()

    def plot_per_class_metrics(self, save_path=None):
        """Plot per-class metrics comparison."""
        metrics = self.compute_all_metrics()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Class Metrics Comparison', fontsize=16, fontweight='bold')

        metric_names = ['IoU', 'Precision', 'Recall', 'F1/Dice']
        metric_keys = ['per_class_iou', 'per_class_precision', 
                       'per_class_recall', 'per_class_dice']

        for idx, (ax, metric_name, metric_key) in enumerate(zip(axes.flat, metric_names, metric_keys)):
            values = metrics[metric_key]
            bars = ax.bar(self.class_names, values, color='steelblue', alpha=0.7)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved: {save_path}")

        return fig

    def save_results(self, output_dir, model_name):
        """
        Save comprehensive results in multiple formats.
        """
        results_dir = os.path.join(output_dir, f'{model_name}_results')
        os.makedirs(results_dir, exist_ok=True)

        # Compute all metrics
        all_metrics = self.compute_all_metrics()

        # 1. Save all metrics as JSON
        metrics_json_path = os.path.join(results_dir, 'all_metrics.json')
        with open(metrics_json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"All metrics saved: {metrics_json_path}")

        # 2. Confusion Matrix
        cm = self.get_confusion_matrix()
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_df.to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

        # Save plots
        self.plot_confusion_matrix(
            os.path.join(results_dir, 'confusion_matrix_normalized.png'),
            normalize=True
        )
        self.plot_confusion_matrix(
            os.path.join(results_dir, 'confusion_matrix_counts.png'),
            normalize=False
        )
        plt.close('all')

        # 3. Per-class metrics plot
        self.plot_per_class_metrics(
            os.path.join(results_dir, 'per_class_metrics.png')
        )
        plt.close('all')

        # 4. Create detailed metrics DataFrame
        metrics_rows = []
        
        # Overall metrics row
        overall_row = {
            'Class': 'Overall',
            'Accuracy': all_metrics['accuracy'],
            'Mean_IoU': all_metrics['mean_iou'],
            'Overall_Dice': all_metrics['overall_dice'],
            'Macro_Precision': all_metrics['macro_precision'],
            'Macro_Recall': all_metrics['macro_recall'],
            'Macro_F1': all_metrics['macro_f1'],
            'Cohen_Kappa': all_metrics['cohen_kappa'],
            'Support': sum(all_metrics[f'{cn}_support'] for cn in self.class_names)
        }
        metrics_rows.append(overall_row)
        
        # Per-class rows
        for i, class_name in enumerate(self.class_names):
            class_row = {
                'Class': class_name,
                'IoU': all_metrics['per_class_iou'][i],
                'Precision': all_metrics['per_class_precision'][i],
                'Recall': all_metrics['per_class_recall'][i],
                'F1_Dice': all_metrics['per_class_dice'][i],
                'Support': all_metrics[f'{class_name}_support']
            }
            metrics_rows.append(class_row)
        
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(results_dir, 'metrics_summary.csv'), index=False)
        print(f"Metrics summary saved: {os.path.join(results_dir, 'metrics_summary.csv')}")

        # 5. Patch-level metrics
        if self.patch_level_metrics:
            patch_df = pd.DataFrame(self.patch_level_metrics)
            patch_df.to_csv(os.path.join(results_dir, 'patch_level_metrics.csv'), index=False)

            summary_stats = patch_df.describe()
            summary_stats.to_csv(os.path.join(results_dir, 'patch_metrics_summary.csv'))

        # 6. Raw data (for further analysis)
        raw_data = {
            'all_predictions': self.all_predictions,
            'all_labels': self.all_labels,
            'class_names': self.class_names,
            'n_classes': self.n_classes,
            'ignore_index': self.ignore_index,
            'all_metrics': all_metrics
        }

        with open(os.path.join(results_dir, 'raw_predictions.pkl'), 'wb') as f:
            pickle.dump(raw_data, f)

        # 7. Summary text report
        self._save_summary_report(results_dir, all_metrics)

        print(f"All results saved to: {results_dir}")
        return results_dir

    def _save_summary_report(self, results_dir, all_metrics):
        """Save a human-readable summary report."""
        with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SEGMENTATION RESULTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy:           {all_metrics['accuracy']:.4f}\n")
            f.write(f"Mean IoU:           {all_metrics['mean_iou']:.4f}\n")
            f.write(f"Overall Dice/F1:    {all_metrics['overall_dice']:.4f}\n")
            f.write(f"Cohen's Kappa:      {all_metrics['cohen_kappa']:.4f}\n\n")

            f.write(f"Macro Precision:    {all_metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall:       {all_metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1:           {all_metrics['macro_f1']:.4f}\n\n")

            # Per-class results
            f.write("=" * 70 + "\n")
            f.write("PER-CLASS RESULTS:\n")
            f.write("=" * 70 + "\n\n")
            
            for class_name in self.class_names:
                f.write(f"{class_name}:\n")
                f.write(f"  IoU:        {all_metrics[f'{class_name}_iou']:.4f}\n")
                f.write(f"  Precision:  {all_metrics[f'{class_name}_precision']:.4f}\n")
                f.write(f"  Recall:     {all_metrics[f'{class_name}_recall']:.4f}\n")
                f.write(f"  F1/Dice:    {all_metrics[f'{class_name}_dice']:.4f}\n")
                f.write(f"  Support:    {all_metrics[f'{class_name}_support']}\n\n")

            # Confusion Matrix
            f.write("=" * 70 + "\n")
            f.write("CONFUSION MATRIX:\n")
            f.write("=" * 70 + "\n")
            cm = all_metrics['confusion_matrix']
            
            f.write("         ")
            for name in self.class_names:
                f.write(f"{name[:10]:>10}")
            f.write("\n")
            
            for i, true_name in enumerate(self.class_names):
                f.write(f"{true_name[:10]:>10}")
                for j in range(len(self.class_names)):
                    f.write(f"{cm[i][j]:>10}")
                f.write("\n")


def predict_with_reconstruction_and_metrics(model, image_link, device=None, batch_size=8,
                                            model_name='prithvi', output_dir='reconstructed_tiles',
                                            labels_path=None, class_names=None):
    """
    Predict on patches, track metrics, and reconstruct full tile as GeoTIFF.
    """
    n_classes = 4
    if not device:
        device = torch.device('cpu')

    # Initialize metrics tracker
    if class_names is None:
        class_names = ['water', 'trees', 'buildings', 'crops']

    metrics_tracker = SegmentationMetricsTracker(
        n_classes=n_classes,
        class_names=class_names,
        ignore_index=255,
        device=device
    )

    # CRITICAL: Must use deterministic mode for reconstruction
    has_labels = labels_path is not None
    test_set = HLSDataset(image_link, labels_path, tile=224, stride=224, random_crop=False, ignore_index=255)

    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    model = model.to(device)
    model.eval()

    print(f'Processing with metrics tracking: {model_name}')

    # Dictionary to store predictions: {patch_index: prediction_array}
    patch_predictions = {}
    patch_index = 0

    with torch.inference_mode():
        for batch_data in test_loader:
            if has_labels:
                image, label = batch_data
                image = image.to(device)
                label = label.to(device)
            else:
                image = batch_data.to(device)
                label = None

            # Get predictions
            logits = model(image)  # B C H W
            preds = logits.argmax(dim=1)  # B H W

            # Update metrics if we have labels
            if has_labels:
                # Get patch information for detailed tracking
                for i in range(image.size(0)):
                    patch_info = {
                        'patch_index': patch_index + i,
                        'batch_size': image.size(0)
                    }
                    # Add window coordinates if available
                    if hasattr(test_set, 'windows') and (patch_index + i) < len(test_set.windows):
                        row, col = test_set.windows[patch_index + i]
                        patch_info['window_row'] = row
                        patch_info['window_col'] = col

                # Update metrics for this batch
                metrics_tracker.update(preds, label, patch_info)

            # Store each prediction with its index
            for batch_idx in range(image.size(0)):
                pred_np = preds[batch_idx].cpu().numpy().astype(np.uint8)
                patch_predictions[patch_index] = pred_np
                patch_index += 1

    # Reconstruct the full tile
    reconstructed_tile = reconstruct_and_save_geotiff(
        patch_predictions=patch_predictions,
        dataset=test_set,
        model_name=model_name,
        output_dir=output_dir
    )

    # Save comprehensive metrics if we have labels
    if has_labels:
        results_dir = metrics_tracker.save_results(output_dir, model_name)
        print(f"Metrics and analysis saved to: {results_dir}")
        return reconstructed_tile, metrics_tracker
    else:
        print("No labels provided - skipping metrics calculation")
        return reconstructed_tile, None


def reconstruct_and_save_geotiff(patch_predictions, dataset, model_name, output_dir):
    """
    Reconstruct full tile from patches and save as GeoTIFF with proper georeferencing.
    """

    # Extract spatial information from dataset (adapted for HLS)
    if hasattr(dataset, 'hls_shape'):
        full_height, full_width = dataset.hls_shape
        transform = dataset.hls_transform
        crs = dataset.hls_crs
    elif hasattr(dataset, 'aligned_height'):
        # Alternative naming convention
        full_height, full_width = dataset.aligned_height, dataset.aligned_width
        transform = dataset.aligned_transform
        crs = dataset.hls_crs
    else:
        raise ValueError("Dataset doesn't have required spatial metadata (hls_shape, hls_transform, hls_crs)")

    print(f"Reconstructing {full_width}x{full_height} pixel tile")

    # Initialize full prediction array
    full_tile = np.full((full_height, full_width), 255, dtype=np.uint8)  # 255 = no data

    # Place each patch back in its original position
    for patch_idx, prediction in patch_predictions.items():
        # Get the original row,col coordinates for this patch
        row, col = dataset.windows[patch_idx]

        # Get patch dimensions (usually 224x224, but could be smaller at edges)
        patch_h, patch_w = prediction.shape

        # Calculate where this patch goes in the full image
        end_row = min(row + patch_h, full_height)
        end_col = min(col + patch_w, full_width)

        # Place the prediction
        full_tile[row:end_row, col:end_col] = prediction[:end_row - row, :end_col - col]

    # Create output filename
    os.makedirs(output_dir, exist_ok=True)
    tile_name = os.path.splitext(os.path.basename(dataset.image_preloc))[0]
    output_file = os.path.join(output_dir, f"{model_name}_{tile_name}_reconstructed.tif")

    # Save as GeoTIFF with full georeferencing
    save_georeferenced_geotiff(
        array=full_tile,
        output_path=output_file,
        transform=transform,
        crs=crs
    )

    print(f"Reconstructed tile saved: {output_file}")
    return full_tile


def save_georeferenced_geotiff(array, output_path, transform, crs,
                               nodata_value=255, compress='lzw'):
    """
    Save numpy array as georeferenced GeoTIFF.
    """

    height, width = array.shape

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256
    ) as dst:
        # Write the data
        dst.write(array, 1)

        # Add metadata
        dst.update_tags(
            DESCRIPTION=f'Prithvi segmentation reconstruction',
            CLASSES='0=water, 1=trees, 2=buildings, 3=crops, 255=nodata',
            MODEL='Prithvi',
            PIXEL_SIZE=f'{abs(transform.a)}m',
            CRS=str(crs)
        )

    print(f"GeoTIFF saved with CRS: {crs}")


def main():
    """
    Main function for multi-city Prithvi model inference with comprehensive metrics tracking.
    """

    # Image paths for all 5 cities
    image_paths = [
        "./Dataset/HLS-2/Orlando/HLS.S30.T17RMM.2024098T155819.v2.0.B02.tif",
        "./Dataset/HLS-2/Seattle/HLS.S30.T10TET.2025159T190909.v2.0.B02.tif",
        "./Dataset/HLS-2/Los Angeles/HLS.S30.T11SLT.2024128T182921.v2.0.B02.tif",
        "./Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif",
        "./Dataset/HLS-2/New York City/HLS.S30.T18TWL.2025279T155029.v2.0.B02.tif",
    ]
    
    # Single label file for all cities
    label_path = "./Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"

    # Extract city names from paths
    city_names = []
    for path in image_paths:
        city_name = path.split('/')[2]  # Gets "Orlando", "Seattle", etc.
        city_names.append(city_name)

    batch_size = 8
    base_output_dir = 'reconstructed_tiles'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Model parameters
    n_classes = 4
    class_names = ['water', 'trees', 'buildings', 'crops']
    model_name = 'prithvi'

    print(f"\n{'=' * 70}")
    print(f"MULTI-CITY INFERENCE - Prithvi 300M Model")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Cities to process: {len(image_paths)}")
    print(f"{'=' * 70}\n")

    """
    ==============================================================================
    MODEL INITIALIZATION (Once for all cities)
    ==============================================================================
    """
    print("Loading Prithvi 300M model...")
    pretrained_model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True)

    model = Prithvi_EO(
        pretrained_model=pretrained_model,
        num_classes=4,
        fpn_blocks=[6, 12, 18, 24],
        scale_factors=[1, 2, 3, 6],
        embed_dim=1024,
        out_channels_feature_map=256,
        FPN_out_channels=256,
        upsampling_scale_list=[4, 2, 1, 0.5],
        u_height=56,
        u_width=56,
    )
    
    # Load trained weights
    model_weights_path = "./prithvi_model/saved_models/Pritvi_300M.pt"
    print(f"Loading trained weights from: {model_weights_path}")
    
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Trained weights loaded successfully!")
    else:
        print(f"WARNING: Model weights not found at {model_weights_path}")
        print("  Using randomly initialized weights (for testing only)")
    
    model = model.to(device)
    model.eval()
    print("Model ready for inference!\n")

    """
    ==============================================================================
    PROCESS EACH CITY
    ==============================================================================
    """
    all_city_results = []
    
    for idx, (image_path, city_name) in enumerate(zip(image_paths, city_names)):
        print(f"\n{'=' * 70}")
        print(f"Processing City {idx + 1}/{len(image_paths)}: {city_name}")
        print(f"{'=' * 70}")
        
        # Create city-specific output directory
        city_output_dir = os.path.join(base_output_dir, city_name)
        os.makedirs(city_output_dir, exist_ok=True)
        
        try:
            # Run inference for this city
            reconstructed_tile, metrics_tracker = predict_with_reconstruction_and_metrics(
                model=model,
                image_link=image_path,
                device=device,
                batch_size=batch_size,
                model_name=model_name,
                output_dir=city_output_dir,
                labels_path=label_path,
                class_names=class_names
            )

            if metrics_tracker is not None:
                all_metrics = metrics_tracker.compute_all_metrics()
                
                print(f"\n{city_name} Results:")
                print(f"  Overall Accuracy: {all_metrics['accuracy']:.4f}")
                print(f"  Mean IoU:         {all_metrics['mean_iou']:.4f}")
                print(f"  Overall Dice:     {all_metrics['overall_dice']:.4f}")
                print(f"  Cohen's Kappa:    {all_metrics['cohen_kappa']:.4f}")
                
                # Store results for summary
                city_result = {
                    'city': city_name,
                    'accuracy': all_metrics['accuracy'],
                    'mean_iou': all_metrics['mean_iou'],
                    'overall_dice': all_metrics['overall_dice'],
                    'cohen_kappa': all_metrics['cohen_kappa'],
                    'macro_precision': all_metrics['macro_precision'],
                    'macro_recall': all_metrics['macro_recall'],
                    'macro_f1': all_metrics['macro_f1'],
                }
                
                # Add per-class F1 scores
                for class_name in class_names:
                    city_result[f'{class_name}_f1'] = all_metrics[f'{class_name}_f1']
                    city_result[f'{class_name}_iou'] = all_metrics[f'{class_name}_iou']
                
                all_city_results.append(city_result)
                
            print(f"{city_name} processing complete!")
            print(f"  Results saved to: {city_output_dir}")
            
        except Exception as e:
            print(f"ERROR processing {city_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    """
    ==============================================================================
    SAVE AGGREGATED RESULTS
    ==============================================================================
    """
    if all_city_results:
        print(f"\n{'=' * 70}")
        print("AGGREGATED RESULTS ACROSS ALL CITIES")
        print(f"{'=' * 70}\n")
        
        # Save summary CSV
        summary_df = pd.DataFrame(all_city_results)
        summary_path = os.path.join(base_output_dir, 'all_cities_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        
        # Print summary table
        print("\nPer-City Performance:")
        print(summary_df[['city', 'accuracy', 'mean_iou', 'overall_dice', 'cohen_kappa']].to_string(index=False))
        
        # Calculate and print overall averages
        print(f"\n{'=' * 70}")
        print("OVERALL AVERAGES:")
        print(f"{'=' * 70}")
        print(f"Mean Accuracy:      {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}")
        print(f"Mean IoU:           {summary_df['mean_iou'].mean():.4f} ± {summary_df['mean_iou'].std():.4f}")
        print(f"Mean Dice/F1:       {summary_df['overall_dice'].mean():.4f} ± {summary_df['overall_dice'].std():.4f}")
        print(f"Mean Cohen's Kappa: {summary_df['cohen_kappa'].mean():.4f} ± {summary_df['cohen_kappa'].std():.4f}")
        print(f"Mean Precision:     {summary_df['macro_precision'].mean():.4f} ± {summary_df['macro_precision'].std():.4f}")
        print(f"Mean Recall:        {summary_df['macro_recall'].mean():.4f} ± {summary_df['macro_recall'].std():.4f}")
        
        print(f"\n{'=' * 70}")
        print("ALL CITIES PROCESSED SUCCESSFULLY!")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
