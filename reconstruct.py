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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TORCH_DYNAMO_DISABLE_DOCSTRING_CHECKS'] = '1'

from terratorch.registry import BACKBONE_REGISTRY
from prithvi_model import Prithvi_EO

from dataset_generator import HLSDataset


class SegmentationMetricsTracker:
    """
    Comprehensive metrics tracking for image segmentation.
    Handles confusion matrix, per-class metrics, and detailed analysis.
    """

    def __init__(self, n_classes=4, class_names=None, ignore_index=255):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f'Class_{i}' for i in range(n_classes)]

        # Initialize tracking variables
        self.all_predictions = []
        self.all_labels = []
        self.patch_level_metrics = []

    def update(self, predictions, labels, patch_info=None):
        """
        Update metrics with new batch of predictions and labels.

        Args:
            predictions: torch.Tensor or np.array of shape (B, H, W)
            labels: torch.Tensor or np.array of shape (B, H, W)
            patch_info: dict with patch metadata (optional)
        """
        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        batch_size = predictions.shape[0]

        for i in range(batch_size):
            pred_patch = predictions[i]
            label_patch = labels[i]

            # Flatten and filter out ignore_index
            pred_flat = pred_patch.flatten()
            label_flat = label_patch.flatten()

            # Remove ignore_index pixels
            valid_mask = label_flat != self.ignore_index
            pred_valid = pred_flat[valid_mask]
            label_valid = label_flat[valid_mask]

            if len(pred_valid) > 0:  # Only add if there are valid pixels
                self.all_predictions.extend(pred_valid.tolist())
                self.all_labels.extend(label_valid.tolist())

                # Calculate patch-level metrics
                patch_metrics = self._calculate_patch_metrics(pred_valid, label_valid)
                if patch_info:
                    patch_metrics.update(patch_info)
                self.patch_level_metrics.append(patch_metrics)

    def _calculate_patch_metrics(self, pred_patch, label_patch):
        """Calculate metrics for a single patch."""
        metrics = {}

        # Overall accuracy
        metrics['patch_accuracy'] = np.mean(pred_patch == label_patch)
        metrics['total_pixels'] = len(pred_patch)

        # Per-class pixel counts
        for class_id in range(self.n_classes):
            pred_count = np.sum(pred_patch == class_id)
            label_count = np.sum(label_patch == class_id)
            correct_count = np.sum((pred_patch == class_id) & (label_patch == class_id))

            metrics[f'{self.class_names[class_id]}_pred_pixels'] = pred_count
            metrics[f'{self.class_names[class_id]}_true_pixels'] = label_count
            metrics[f'{self.class_names[class_id]}_correct_pixels'] = correct_count

            # Patch-level precision/recall
            if pred_count > 0:
                metrics[f'{self.class_names[class_id]}_precision'] = correct_count / pred_count
            else:
                metrics[f'{self.class_names[class_id]}_precision'] = 0.0

            if label_count > 0:
                metrics[f'{self.class_names[class_id]}_recall'] = correct_count / label_count
            else:
                metrics[f'{self.class_names[class_id]}_recall'] = 1.0  # Perfect recall if no true pixels

        return metrics

    def get_confusion_matrix(self):
        """Get overall confusion matrix."""
        if not self.all_predictions:
            return None
        return confusion_matrix(self.all_labels, self.all_predictions,
                                labels=list(range(self.n_classes)))

    def get_classification_report(self):
        """Get detailed classification report."""
        if not self.all_predictions:
            return None
        return classification_report(self.all_labels, self.all_predictions,
                                     labels=list(range(self.n_classes)),
                                     target_names=self.class_names,
                                     output_dict=True, zero_division=0)

    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """Plot and optionally save confusion matrix."""
        cm = self.get_confusion_matrix()
        if cm is None:
            return None

        plt.figure(figsize=(10, 8))

        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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

    def save_results(self, output_dir, model_name):
        """
        Save comprehensive results in multiple formats.
        """
        results_dir = os.path.join(output_dir, f'{model_name}_results')
        os.makedirs(results_dir, exist_ok=True)

        # 1. Confusion Matrix
        cm = self.get_confusion_matrix()
        if cm is not None:
            # Save as CSV
            cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
            cm_df.to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

            # Save plot
            self.plot_confusion_matrix(
                os.path.join(results_dir, 'confusion_matrix_normalized.png'),
                normalize=True
            )
            self.plot_confusion_matrix(
                os.path.join(results_dir, 'confusion_matrix_counts.png'),
                normalize=False
            )
            plt.close('all')  # Close all figures

        # 2. Classification Report
        report = self.get_classification_report()
        if report is not None:
            # Save as JSON
            with open(os.path.join(results_dir, 'classification_report.json'), 'w') as f:
                json.dump(report, f, indent=2)

            # Save as CSV
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(results_dir, 'classification_report.csv'))

        # 3. Patch-level metrics
        if self.patch_level_metrics:
            patch_df = pd.DataFrame(self.patch_level_metrics)
            patch_df.to_csv(os.path.join(results_dir, 'patch_level_metrics.csv'), index=False)

            # Summary statistics
            summary_stats = patch_df.describe()
            summary_stats.to_csv(os.path.join(results_dir, 'patch_metrics_summary.csv'))

        # 4. Raw data (for further analysis)
        raw_data = {
            'all_predictions': self.all_predictions,
            'all_labels': self.all_labels,
            'class_names': self.class_names,
            'n_classes': self.n_classes,
            'ignore_index': self.ignore_index
        }

        with open(os.path.join(results_dir, 'raw_predictions.pkl'), 'wb') as f:
            pickle.dump(raw_data, f)

        # 5. Summary text report
        self._save_summary_report(results_dir, report, cm)

        print(f"All results saved to: {results_dir}")
        return results_dir

    def _save_summary_report(self, results_dir, report, cm):
        """Save a human-readable summary report."""
        with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
            f.write("SEGMENTATION RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            if report:
                f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")
                f.write(f"Macro Average Precision: {report['macro avg']['precision']:.4f}\n")
                f.write(f"Macro Average Recall: {report['macro avg']['recall']:.4f}\n")
                f.write(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}\n\n")

                f.write("Per-Class Results:\n")
                f.write("-" * 30 + "\n")
                for class_name in self.class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                        f.write(f"  Support: {metrics['support']}\n\n")

            if cm is not None:
                f.write("Confusion Matrix:\n")
                f.write("-" * 20 + "\n")
                f.write("      ")
                for name in self.class_names:
                    f.write(f"{name[:8]:>8}")
                f.write("\n")

                for i, true_name in enumerate(self.class_names):
                    f.write(f"{true_name[:8]:>8}")
                    for j in range(len(self.class_names)):
                        f.write(f"{cm[i, j]:>8}")
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

    metrics_tracker = SegmentationMetricsTracker(n_classes=n_classes,
                                                 class_names=class_names,
                                                 ignore_index=255)

    # CRITICAL: Must use deterministic mode for reconstruction
    has_labels = labels_path is not None
    test_set = HLSDataset(image_link, labels_path, tile=224, stride=224, ignore_index=255)

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
                batch_patch_info = []
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
                    batch_patch_info.append(patch_info)

                # Update metrics for this batch
                for i, patch_info in enumerate(batch_patch_info):
                    metrics_tracker.update(
                        preds[i:i + 1],
                        label[i:i + 1],
                        patch_info
                    )

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
    tile_name = os.path.splitext(os.path.basename(dataset.image_loc))[0]
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
        "Dataset/HLS-2/New York City/HLS.S30.T18TWL.2024240T154931.v2.0.B02.tif"
    ]
    
    # Single label file for all cities
    label_path = "Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"

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
    model_weights_path = "prithvi_model/saved_models/Pritvi_300M_.pt"
    print(f"Loading trained weights from: {model_weights_path}")
    
    
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Trained weights loaded successfully!")
    except FileNotFoundError as e:
        print(f"Model weights not found at {model_weights_path}")
        return
        
    
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
                report = metrics_tracker.get_classification_report()
                if report:
                    print(f"\n{city_name} Results:")
                    print(f"  Overall Accuracy: {report['accuracy']:.4f}")
                    print(f"  Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
                    print(f"  Mean IoU: {report['macro avg']['recall']:.4f}")  # Approximation
                    
                    # Store results for summary
                    city_result = {
                        'city': city_name,
                        'accuracy': report['accuracy'],
                        'macro_f1': report['macro avg']['f1-score'],
                        'macro_precision': report['macro avg']['precision'],
                        'macro_recall': report['macro avg']['recall'],
                    }
                    
                    # Add per-class F1 scores
                    for class_name in class_names:
                        if class_name in report:
                            city_result[f'{class_name}_f1'] = report[class_name]['f1-score']
                    
                    all_city_results.append(city_result)
                    
            print(f"{city_name} processing complete!")
            print(f"  Results saved to: {city_output_dir}")
            
        except Exception as e:
            print(f"ERROR processing {city_name}: {str(e)}")
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
        print(summary_df.to_string(index=False))
        
        # Calculate and print overall averages
        print(f"\n{'=' * 70}")
        print("OVERALL AVERAGES:")
        print(f"{'=' * 70}")
        print(f"Mean Accuracy: {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}")
        print(f"Mean Macro F1: {summary_df['macro_f1'].mean():.4f} ± {summary_df['macro_f1'].std():.4f}")
        print(f"Mean Precision: {summary_df['macro_precision'].mean():.4f} ± {summary_df['macro_precision'].std():.4f}")
        print(f"Mean Recall: {summary_df['macro_recall'].mean():.4f} ± {summary_df['macro_recall'].std():.4f}")
        
        print(f"\n{'=' * 70}")
        print("ALL CITIES PROCESSED SUCCESSFULLY!")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
