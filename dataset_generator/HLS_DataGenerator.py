# -*- coding: utf-8 -*-
"""
Unified HLSDataset: Handles both training (with labels) and inference (without labels)
"""

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from shapely.geometry import box
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.transform import Affine


class HLSDataset(Dataset):
    """
    Unified PyTorch Dataset for HLS imagery supporting:
    - Training mode: with labels (label_loc provided)
    - Inference mode: without labels (label_loc=None)
    """

    def __init__(self, image_loc, label_loc=None, tile=224, stride=224, 
                 ignore_index=255, verbose=False):
        """
        Parameters
        ----------
        image_loc : str
            Path to HLS band (e.g., B02)
        label_loc : str or None
            Path to label raster. If None, runs in inference mode (no labels)
        tile : int
            Patch size in pixels
        stride : int
            Step size for sliding window
        ignore_index : int
            Value for invalid/ignored pixels in labels
        verbose : bool
            Print debug information
        """
        self.image_loc = image_loc
        self.label_loc = label_loc
        self.tile = tile
        self.stride = stride
        self.ignore_index = ignore_index
        self.verbose = verbose
        
        # Determine mode
        self.inference_mode = (label_loc is None)
        
        if self.inference_mode:
            print("Running in INFERENCE MODE (no labels)")
            self._setup_inference_grid()
        else:
            print("Running in TRAINING MODE (with labels)")
            self._setup_aligned_grid()

        # HLS band names
        self.band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
        
        # Derive all band paths
        self.all_band_locations = [
            self.image_loc[:-7] + band_name + self.image_loc[-4:] 
            for band_name in self.band_names
        ]

        # Build patch windows
        self.windows = self._build_windows()
        
        print(f"Dataset initialized: {len(self.windows)} patches available")

    # ========== INFERENCE MODE SETUP ==========
    
    def _setup_inference_grid(self):
        """Setup grid parameters when no labels are provided (inference mode)."""
        with rasterio.open(self.image_loc) as src:
            self.hls_transform = src.transform
            self.hls_bounds = src.bounds
            self.hls_crs = src.crs
            self.hls_shape = src.shape

        # In inference mode, use the entire HLS image extent
        pixel_size = self.hls_transform.a
        
        self.aligned_bounds = self.hls_bounds
        self.aligned_width = self.hls_shape[1]
        self.aligned_height = self.hls_shape[0]
        self.aligned_transform = self.hls_transform

        if self.verbose:
            print(f"[HLS IMAGE] bounds: {self.hls_bounds}")
            print(f"Grid: {self.aligned_width}x{self.aligned_height} pixels")
            print(f"Pixel size: {pixel_size}m")

    # ========== TRAINING MODE SETUP ==========
    
    def _setup_aligned_grid(self):
        """Setup aligned grid when labels are provided (training mode)."""
        # Load HLS image metadata
        with rasterio.open(self.image_loc) as src_image:
            self.hls_transform = src_image.transform
            self.hls_bounds = src_image.bounds
            self.hls_crs = src_image.crs
            self.hls_shape = src_image.shape

        # Load label metadata
        with rasterio.open(self.label_loc) as src_labels:
            self.label_transform = src_labels.transform
            original_label_bounds = src_labels.bounds
            self.label_crs = src_labels.crs
            self.label_shape = src_labels.shape

        if self.verbose:
            print(f"[HLS IMAGE] bounds: {self.hls_bounds}")
            print(f"[LABEL] bounds: {original_label_bounds}")

        # Transform label bounds if CRS mismatch
        if self.label_crs != self.hls_crs:
            from rasterio.warp import transform_bounds
            self.label_bounds = transform_bounds(
                self.label_crs,
                self.hls_crs,
                *original_label_bounds
            )
            if self.verbose:
                print(f"[LABEL TRANSFORMED] bounds: {self.label_bounds}")
        else:
            self.label_bounds = original_label_bounds

        # Compute intersection
        hls_box = box(*self.hls_bounds)
        label_box = box(*self.label_bounds)
        intersection = hls_box.intersection(label_box)

        assert not intersection.is_empty, \
            "No overlap between image and labels - check your data alignment"

        self.intersection_bounds = intersection.bounds

        # Snap to pixel grid
        pixel_size = self.hls_transform.a
        
        min_x = np.floor((self.intersection_bounds[0] - self.hls_bounds[0]) / pixel_size) * pixel_size + self.hls_bounds[0]
        min_y = np.floor((self.intersection_bounds[1] - self.hls_bounds[1]) / pixel_size) * pixel_size + self.hls_bounds[1]
        max_x = np.ceil((self.intersection_bounds[2] - self.hls_bounds[0]) / pixel_size) * pixel_size + self.hls_bounds[0]
        max_y = np.ceil((self.intersection_bounds[3] - self.hls_bounds[1]) / pixel_size) * pixel_size + self.hls_bounds[1]

        self.aligned_bounds = (min_x, min_y, max_x, max_y)
        self.aligned_width = int((max_x - min_x) / pixel_size)
        self.aligned_height = int((max_y - min_y) / pixel_size)

        self.aligned_transform = transform_from_bounds(
            *self.aligned_bounds, self.aligned_width, self.aligned_height
        )

        print(f"Aligned grid: {self.aligned_width}x{self.aligned_height} pixels")
        print(f"Pixel size: {pixel_size}m")

    # ========== WINDOW GENERATION ==========
    
    def _build_windows(self):
        """Build list of (row, col) patch coordinates."""
        windows = []
        for r in range(0, self.aligned_height - self.tile + 1, self.stride):
            for c in range(0, self.aligned_width - self.tile + 1, self.stride):
                windows.append((r, c))
        return windows

    # ========== DATA LOADING ==========
    
    def __getitem__(self, idx):
        """
        Returns:
        - Training mode: (image, label) tuple
        - Inference mode: image only
        """
        row, col = self.windows[idx]

        img = self._load_HLS_image(row, col, self.tile)
        img = torch.from_numpy(img).float()

        if self.inference_mode:
            # Return only image for inference
            return img
        else:
            # Return image and label for training
            lbl = self._load_labels_patch(row, col, self.tile)
            lbl = torch.from_numpy(lbl).long()
            return img, lbl

    def _load_HLS_image(self, row, col, size):
        """Load multi-band HLS image patch."""
        patch_bounds = (
            self.aligned_bounds[0] + col * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + (row + size) * self.aligned_transform.e,
            self.aligned_bounds[0] + (col + size) * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + row * self.aligned_transform.e
        )

        bands = []
        for band_location in self.all_band_locations:
            with rasterio.open(band_location) as src:
                window = from_bounds(*patch_bounds, transform=src.transform)
                band_data = src.read(1, window=window).astype(np.float32)

                # Handle edge cases
                if band_data.shape != (size, size):
                    padded = np.zeros((size, size), dtype=np.float32)
                    h, w = min(band_data.shape[0], size), min(band_data.shape[1], size)
                    padded[:h, :w] = band_data[:h, :w]
                    band_data = padded

                bands.append(band_data)

        stacked = np.stack(bands)
        stacked = stacked / 10000.0  # Normalize reflectance values

        return stacked

    def _load_labels_patch(self, row, col, size):
        """Load label patch (training mode only)."""
        if self.inference_mode:
            raise RuntimeError("Cannot load labels in inference mode")

        patch_bounds = (
            self.aligned_bounds[0] + col * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + (row + size) * self.aligned_transform.e,
            self.aligned_bounds[0] + (col + size) * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + row * self.aligned_transform.e
        )

        with rasterio.open(self.label_loc) as src:
            if src.crs != self.hls_crs:
                # Reproject if CRS differs
                from rasterio.warp import transform_bounds
                patch_bounds_label_crs = transform_bounds(
                    self.hls_crs, src.crs, *patch_bounds
                )

                window = from_bounds(*patch_bounds_label_crs, transform=src.transform)
                lbl_data = src.read(1, window=window)

                label_patch = np.full((size, size), self.ignore_index, dtype=np.uint8)

                reproject(
                    source=lbl_data,
                    destination=label_patch,
                    src_transform=rasterio.windows.transform(window, src.transform),
                    src_crs=src.crs,
                    dst_transform=self.aligned_transform * Affine.translation(col, row),
                    dst_crs=self.hls_crs,
                    resampling=Resampling.nearest,
                )
            else:
                # Same CRS, direct read
                window = from_bounds(*patch_bounds, transform=src.transform)
                label_patch = src.read(1, window=window)

                if label_patch.shape != (size, size):
                    padded = np.full((size, size), self.ignore_index, dtype=np.uint8)
                    h, w = min(label_patch.shape[0], size), min(label_patch.shape[1], size)
                    padded[:h, :w] = label_patch[:h, :w]
                    label_patch = padded

            label_patch = self._remap_labels(label_patch)

            return label_patch

    def _remap_labels(self, label_patch):
        """Remap NLCD classes to simplified categories."""
        remapped = np.full_like(label_patch, 255, dtype=np.uint8)

        remapped[label_patch == 11] = 0  # Water
        remapped[np.isin(label_patch, [41, 42, 43, 52, 90])] = 1  # Trees
        remapped[np.isin(label_patch, [22, 23, 24])] = 2  # Built-up
        remapped[np.isin(label_patch, [21, 71, 72, 73, 74, 81, 82, 95])] = 3  # Grassland

        return remapped

    def __len__(self):
        return len(self.windows)

    # ========== VISUALIZATION ==========
    
    def plot_sample(self, idx, figsize=(15, 5)):
        """
        Visualize a sample.
        - Training mode: shows RGB, labels, and overlay
        - Inference mode: shows RGB only
        """
        data = self[idx]
        
        if self.inference_mode:
            img = data
            self._plot_inference(img, idx, figsize)
        else:
            img, lbl = data
            self._plot_training(img, lbl, idx, figsize)

    def _plot_inference(self, img, idx, figsize):
        """Plot inference sample (image only)."""
        img_np = img.numpy()

        rgb = np.stack([img_np[2], img_np[1], img_np[0]], axis=-1)
        
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
            rgb_norm[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(rgb_norm)
        ax.set_title(f'RGB Image (Patch {idx})')
        ax.axis('off')

        print(f"\nPatch {idx} (Inference Mode):")
        print(f"Image shape: {img_np.shape}")
        print(f"Value range: [{img_np.min():.4f}, {img_np.max():.4f}]")

        plt.tight_layout()
        plt.show()

    def _plot_training(self, img, lbl, idx, figsize):
        """Plot training sample (image + labels + overlay)."""
        img_np = img.numpy()
        lbl_np = lbl.numpy()

        rgb = np.stack([img_np[2], img_np[1], img_np[0]], axis=-1)
        
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
            rgb_norm[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].imshow(rgb_norm)
        axes[0].set_title(f'RGB Image (Patch {idx})')
        axes[0].axis('off')

        colors = ['blue', 'darkgreen', 'red', '#39FF14', 'white']
        cmap = ListedColormap(colors)
        lbl_masked = np.ma.masked_equal(lbl_np, 255)

        im = axes[1].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[1].set_title('Labels')
        axes[1].axis('off')

        cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Water', 'Trees', 'Built-up', 'Grassland'])

        axes[2].imshow(rgb_norm)
        axes[2].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, alpha=0.5, interpolation='nearest')
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        unique, counts = np.unique(lbl_np, return_counts=True)
        print(f"\nPatch {idx} statistics:")
        print(f"Image shape: {img_np.shape}")
        print(f"Label shape: {lbl_np.shape}")
        print(f"Label distribution:")
        class_names = {0: 'Water', 1: 'Trees', 2: 'Built-up', 3: 'Grassland', 255: 'Ignore'}
        for val, count in zip(unique, counts):
            pct = count / lbl_np.size * 100
            print(f"  {class_names.get(val, val)}: {count} pixels ({pct:.1f}%)")

        plt.tight_layout()
        plt.show()

    def get_patch_info(self, idx):
        """Get geographic information for a patch."""
        row, col = self.windows[idx]
        
        min_x = self.aligned_bounds[0] + col * abs(self.aligned_transform.a)
        max_x = self.aligned_bounds[0] + (col + self.tile) * abs(self.aligned_transform.a)
        min_y = self.aligned_bounds[3] + (row + self.tile) * self.aligned_transform.e
        max_y = self.aligned_bounds[3] + row * self.aligned_transform.e
        
        return {
            'patch_idx': idx,
            'pixel_coords': (row, col),
            'geo_bounds': (min_x, min_y, max_x, max_y),
            'mode': 'inference' if self.inference_mode else 'training'
        }


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    
    # Example 1: Training mode (with labels)
    print("=" * 80)
    print("TRAINING MODE EXAMPLE")
    print("=" * 80)
    
    train_dataset = HLSDataset(
        image_loc="../Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif",
        label_loc="../Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif",
        tile=224,
        stride=224,
        verbose=True
    )
    
    print(f"\nTotal patches: {len(train_dataset)}")
    
    # Get a sample (returns image and label)
    img, lbl = train_dataset[0]
    print(f"Image shape: {img.shape}, Label shape: {lbl.shape}")
    
    # Visualize
    train_dataset.plot_sample(0)
    
    print("\n" + "=" * 80)
    print("INFERENCE MODE EXAMPLE")
    print("=" * 80)
    
    # Example 2: Inference mode (no labels)
    inference_dataset = HLSDataset(
        image_loc="../Dataset/HLS-2/Seattle/HLS.S30.T10TET.2025159T190909.v2.0.B02.tif",
        label_loc=None,  # No labels!
        tile=224,
        stride=224,
        verbose=True
    )
    
    print(f"\nTotal patches: {len(inference_dataset)}")
    
    # Get a sample (returns only image)
    img = inference_dataset[0]
    print(f"Image shape: {img.shape}")
    
    # Visualize
    inference_dataset.plot_sample(0)