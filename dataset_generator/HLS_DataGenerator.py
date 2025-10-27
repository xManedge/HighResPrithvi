# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:02:40 2025

@author: BCC
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

"""
===============================================================================
HLSDataset: Custom PyTorch Dataset for Harmonized Landsat-Sentinel (HLS) data
===============================================================================

This class handles:
- Reading multi-band HLS satellite imagery (Sentinel-2 derived)
- Aligning label rasters (e.g., NLCD) with imagery in a consistent grid
- Dividing the data into patches (tiles) for training segmentation models
- Performing reprojection when label CRS differs from HLS CRS
- Ensuring all patch dimensions align to pixel grid spacing

Mathematically:
Let:
    - (x, y) be geographic coordinates in map space
    - (r, c) be pixel indices in image space
    - T be the affine transform mapping pixel → map coordinates:
          [x]   [a  b  c] [c]
          [y] = [d  e  f] [r]
          [1]   [0  0  1] [1]

    For HLS (30m resolution), `a` ≈ 30 and `e` ≈ -30 (since y decreases downward).
This transform ensures spatial alignment between imagery and label rasters.
"""


class HLSDataset(Dataset):

    def __init__(self, image_loc, label_loc, tile=224, stride=224, ignore_index=255, verbose=False):
        """
        Initializes dataset parameters and computes aligned grids.
        """
        self.image_loc = image_loc  # Path to Sentinel-2 band (e.g., B02)
        self.label_loc = label_loc  # Path to label raster (e.g., NLCD)
        self.tile = tile  # Patch height/width in pixels
        self.stride = stride  # Step between successive patches
        self.ignore_index = ignore_index  # Label mask for invalid pixels
        self.verbose = verbose

        # Prepare coordinate alignment and transformations
        self._setup_aligned_grid()

        # Sentinel-2 bands used (Blue to SWIR)
        self.band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']

        # Derive full band paths (replacing B02 with other band names)
        self.all_band_locations = [self.image_loc[:-7] + band_name + self.image_loc[-4:] for band_name in
                                   self.band_names]

        # Build list of patch starting coordinates (row, col)
        self.windows = self._build_windows()

    """
    ---------------------------------------------------------------------------
    _setup_aligned_grid()
    ---------------------------------------------------------------------------
    Aligns the label raster and the HLS image grid to ensure perfect spatial overlap.

    Steps:
    1. Reads geospatial metadata (CRS, transform, bounds) from image and label.
    2. Transforms label bounds to HLS CRS if necessary.
    3. Computes the intersection area between HLS and label extents.
    4. Snaps the intersection bounds to HLS pixel grid boundaries.

    Mathematically:
        Aligned width  = (max_x - min_x) / pixel_size
        Aligned height = (max_y - min_y) / pixel_size
    ensuring integer pixel coverage.
    """

    def _setup_aligned_grid(self):
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
            print(
                f"[HLS IMAGE]\tmin_x:\t{self.hls_bounds.left}\tmin_y:\t{self.hls_bounds.bottom}\tmax_x:\t{self.hls_bounds.right}\tmax_y:\t{self.hls_bounds.top}")
            print('\n')
            print(
                f"[LABEL]\tmin_x:\t{original_label_bounds.left}\tmin_y:\t{original_label_bounds.bottom}\tmax_x:\t{original_label_bounds.right}\tmax_y:\t{original_label_bounds.top}")

        # Transform label bounds if CRS mismatch
        if self.label_crs != self.hls_crs:
            from rasterio.warp import transform_bounds
            self.label_bounds = transform_bounds(
                self.label_crs,
                self.hls_crs,
                *original_label_bounds
            )
            if self.verbose:
                print(
                    f"[LABEL TRANSFORMED]\tmin_x:\t{self.label_bounds[0]}\tmin_y:\t{self.label_bounds[1]}\tmax_x:\t{self.label_bounds[2]}\tmax_y:\t{self.label_bounds[3]}")
        else:
            self.label_bounds = original_label_bounds

        # Store transform for label CRS reference
        self.label_transform = src_labels.transform

        # Compute geometric intersection (in HLS CRS)
        hls_box = box(*self.hls_bounds)
        label_box = box(*self.label_bounds)
        intersection = hls_box.intersection(label_box)

        # Intersection must exist, else datasets are misaligned
        assert not intersection.is_empty, "Intersection is empty, are you sure your ground truth and your images match"

        self.intersection_bounds = intersection.bounds

        if self.verbose:
            print("intersection bounds: ", self.intersection_bounds)

        # Pixel size (for HLS 30m)
        pixel_size = self.hls_transform.a

        """
        Snapping step:
        Adjust bounds so that (min_x, max_x, min_y, max_y) fall exactly
        on pixel grid lines defined by the affine transform.
        This ensures tiles map cleanly to pixel coordinates without subpixel offsets.
        """
        min_x = np.floor((self.intersection_bounds[0] - self.hls_bounds[0]) / pixel_size) * pixel_size + \
                self.hls_bounds[0]
        min_y = np.floor((self.intersection_bounds[1] - self.hls_bounds[1]) / pixel_size) * pixel_size + \
                self.hls_bounds[1]
        max_x = np.ceil((self.intersection_bounds[2] - self.hls_bounds[0]) / pixel_size) * pixel_size + self.hls_bounds[
            0]
        max_y = np.ceil((self.intersection_bounds[3] - self.hls_bounds[1]) / pixel_size) * pixel_size + self.hls_bounds[
            1]

        if self.verbose:
            print(f"min_x:\t{min_x}\tmin_y:\t{min_y}\tmax_x:\t{max_x}\tmax_y:\t{max_y}")

        self.aligned_bounds = (min_x, min_y, max_x, max_y)

        # Compute pixel dimensions
        self.aligned_width = int((max_x - min_x) / pixel_size)
        self.aligned_height = int((max_y - min_y) / pixel_size)

        # Build affine transform for aligned grid
        self.aligned_transform = transform_from_bounds(
            *self.aligned_bounds, self.aligned_width, self.aligned_height
        )

        print(f"Aligned grid: {self.aligned_width}x{self.aligned_height} pixels")
        print(f"Pixel size: {pixel_size}m")

    """
    ---------------------------------------------------------------------------
    _build_windows()
    ---------------------------------------------------------------------------
    Creates a list of (row, col) top-left pixel coordinates defining each
    patch window to be extracted from the aligned raster.

    Each patch is of dimension:
        (tile x tile) pixels
    and windows slide by `stride` pixels.
    """

    def _build_windows(self):
        windows = []

        for r in range(0, self.aligned_height - self.tile + 1, self.stride):
            for c in range(0, self.aligned_width - self.tile + 1, self.stride):
                windows.append((r, c))

        return windows

    """
    ---------------------------------------------------------------------------
    __getitem__()
    ---------------------------------------------------------------------------
    Retrieves a single image-label patch pair.

    Returns:
        img : Tensor of shape (6, tile, tile)
        lbl : Tensor of shape (tile, tile)
    """

    def __getitem__(self, idx):
        row, col = self.windows[idx]

        img = self._load_HLS_image(row, col, self.tile)
        lbl = self._load_labels_patch(row, col, self.tile)

        if self.verbose:
            print(f"[LOADED IMAGE]\tSource:\t{self.image_loc}")
            print(f"\tPatch index:\t{idx}\t(Row: {row}, Col: {col})")
            print(f"\tImage shape:\t{img.shape}\tLabel shape:\t{lbl.shape}")
            print(f"\tImage min:\t{img.min():.4f}\tmax:\t{img.max():.4f}\tmean:\t{img.mean():.4f}")
            print(f"\tLabel unique values:\t{np.unique(lbl)}")

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    """
    ---------------------------------------------------------------------------
    _load_HLS_image()
    ---------------------------------------------------------------------------
    Extracts a multi-band image patch from the aligned HLS imagery.

    Each patch is read band-by-band and stacked into a tensor:
        Shape = (6, tile, tile)
    """

    def _load_HLS_image(self, row, col, size):
        # Compute bounding box of patch in map coordinates
        patch_bounds = (
            self.aligned_bounds[0] + col * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + (row + size) * self.aligned_transform.e,  # e is negative
            self.aligned_bounds[0] + (col + size) * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + row * self.aligned_transform.e
        )

        bands = []
        for band_location in self.all_band_locations:
            with rasterio.open(band_location) as src:
                window = from_bounds(*patch_bounds, transform=src.transform)
                band_data = src.read(1, window=window).astype(np.float32)

                # Handle edge patches smaller than tile size
                if band_data.shape != (size, size):
                    padded = np.zeros((size, size), dtype=np.float32)
                    h, w = min(band_data.shape[0], size), min(band_data.shape[1], size)
                    padded[:h, :w] = band_data[:h, :w]
                    band_data = padded

                bands.append(band_data)

        stacked = np.stack(bands)

        return stacked

    """
    ---------------------------------------------------------------------------
    _load_labels_patch()
    ---------------------------------------------------------------------------
    Extracts the label patch aligned to the same spatial window as the image.
    Handles reprojection if the label CRS differs from the HLS CRS.

    Output shape:
        (tile, tile)
    """

    def _load_labels_patch(self, row, col, size):
        patch_bounds = (
            self.aligned_bounds[0] + col * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + (row + size) * self.aligned_transform.e,
            self.aligned_bounds[0] + (col + size) * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + row * self.aligned_transform.e
        )

        with rasterio.open(self.label_loc) as src:
            if src.crs != self.hls_crs:
                # If CRS differs, reproject label window
                from rasterio.warp import transform_bounds
                patch_bounds_label_crs = transform_bounds(
                    self.hls_crs,
                    src.crs,
                    *patch_bounds
                )

                window = from_bounds(*patch_bounds_label_crs, transform=src.transform)

                lbl_data = src.read(1, window=window)

                # Initialize empty patch with ignore index
                label_patch = np.full((size, size), self.ignore_index, dtype=np.uint8)

                # Reproject into HLS CRS
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

                # Handle edge size mismatch
                if label_patch.shape != (size, size):
                    padded = np.full((size, size), self.ignore_index, dtype=np.uint8)
                    h, w = min(label_patch.shape[0], size), min(label_patch.shape[1], size)
                    padded[:h, :w] = label_patch[:h, :w]
                    label_patch = padded

            # Simplify land cover classes
            label_patch = self._remap_labels(label_patch)

            return label_patch

    """
    ---------------------------------------------------------------------------
    _remap_labels()
    ---------------------------------------------------------------------------
    Maps complex NLCD class codes into 4 semantic categories:
        0: Water
        1: Forest/Trees
        2: Built-up
        3: Grassland/Rangeland
        255: Ignore
    """

    def _remap_labels(self, label_patch):
        remapped = np.full_like(label_patch, 255, dtype=np.uint8)

        # Water
        remapped[label_patch == 11] = 0

        # Trees/Forest (NLCD codes 41–43, 52, 90)
        remapped[np.isin(label_patch, [41, 42, 43, 52, 90])] = 1

        # Built-up areas
        remapped[np.isin(label_patch, [22, 23, 24])] = 2

        # Grasslands/Rangelands
        remapped[np.isin(label_patch, [21, 71, 72, 73, 74, 81, 82, 95])] = 3

        return remapped

    """
    ---------------------------------------------------------------------------
    __len__()
    ---------------------------------------------------------------------------
    Returns total number of patch windows available in the dataset.
    """

    def __len__(self):
        return len(self.windows)

    """
    ---------------------------------------------------------------------------
    plot_overlay()
    ---------------------------------------------------------------------------
    Visualization utility to show:
        - RGB image composite
        - Corresponding label map
        - Overlay of labels on RGB

    Provides quick inspection of alignment and label integrity.
    """

    def plot_overlay(self, idx, figsize=(15, 5)):
        img, lbl = self[idx]

        img_np = img.numpy()
        lbl_np = lbl.numpy()

        # True color composite (B04-R, B03-G, B02-B)
        rgb = np.stack([
            img_np[2],  # Red
            img_np[1],  # Green
            img_np[0],  # Blue
        ], axis=-1)

        # Normalize brightness using 2–98 percentile stretch
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
            rgb_norm[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)

        # Plot results
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

        # Print class statistics
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

        return fig
'''

import os
from pathlib import Path

def plot_overlay_and_save(dataset, idx, city_name, output_dir='./dataset_gen_output'):
    """
    Create and save an overlay plot of the satellite image and label.
    
    Args:
        dataset: HLSDataset instance
        idx: Index of the patch to visualize
        city_name: Name of the city for filename
        output_dir: Directory to save the output
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get image and label
    img, lbl = dataset[idx]
    
    img_np = img.numpy()
    lbl_np = lbl.numpy()
    
    # True color composite (B04-R, B03-G, B02-B)
    rgb = np.stack([
        img_np[2],  # Red
        img_np[1],  # Green
        img_np[0],  # Blue
    ], axis=-1)
    
    # Normalize brightness using 2–98 percentile stretch
    rgb_norm = np.zeros_like(rgb)
    for i in range(3):
        p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
        rgb_norm[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RGB image
    axes[0].imshow(rgb_norm)
    axes[0].set_title(f'RGB Image - {city_name} (Patch {idx})')
    axes[0].axis('off')
    
    # Plot labels
    colors = ['blue', 'darkgreen', 'red', '#39FF14', 'white']
    cmap = ListedColormap(colors)
    lbl_masked = np.ma.masked_equal(lbl_np, 255)
    
    im = axes[1].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
    axes[1].set_title('Labels')
    axes[1].axis('off')
    
    cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Water', 'Trees', 'Built-up', 'Grassland'])
    
    # Plot overlay
    axes[2].imshow(rgb_norm)
    axes[2].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, alpha=0.5, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{city_name}_{idx}.jpeg')
    plt.savefig(output_path, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    
    # Print statistics
    unique, counts = np.unique(lbl_np, return_counts=True)
    print(f"Patch {idx} statistics for {city_name}:")
    print(f"  Image shape: {img_np.shape}")
    print(f"  Label shape: {lbl_np.shape}")
    class_names = {0: 'Water', 1: 'Trees', 2: 'Built-up', 3: 'Grassland', 255: 'Ignore'}
    for val, count in zip(unique, counts):
        pct = count / lbl_np.size * 100
        print(f"  {class_names.get(val, val)}: {count} pixels ({pct:.1f}%)")
    print()


def main():
    """
    Main function to test HLSDataset with multiple cities and save overlay plots.
    """
    # Define image paths for different cities
    image_paths = [
        "../Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif",
        "../Dataset/HLS-2/Seattle/HLS.S30.T10TET.2025159T190909.v2.0.B02.tif",
        "../Dataset/HLS-2/Los Angeles/HLS.S30.T11SLT.2024128T182921.v2.0.B02.tif",
        "../Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif",
    ]
    
    # City names corresponding to each path
    city_names = ['Orlando', 'Seattle', 'Los_Angeles', 'Chicago']
    
    # Label path (same for all regions)
    label_path = "../Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"
    
    # Output directory
    output_dir = '../dataset_gen_output'
    
    # Test parameters
    tile_size = 224
    stride = 224
    num_samples_per_city = 3  # Number of patches to visualize per city
    
    print("=" * 80)
    print("Testing HLSDataset with multiple cities")
    print("=" * 80)
    
    # Process each city
    for image_path, city_name in zip(image_paths, city_names):
        print(f"\n{'=' * 80}")
        print(f"Processing: {city_name}")
        print(f"{'=' * 80}\n")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"WARNING: Image file not found: {image_path}")
            print(f"Skipping {city_name}...\n")
            continue
        
        # Initialize dataset
        try:
            dataset = HLSDataset(
                image_loc=image_path,
                label_loc=label_path,
                tile=tile_size,
                stride=stride,
                verbose=True
            )
            
            print(f"\nDataset initialized successfully!")
            print(f"Total patches available: {len(dataset)}")
            
            # Determine how many samples to process
            num_samples = min(num_samples_per_city, len(dataset))
            
            # Generate and save overlay plots
            for i in range(num_samples):
                print(f"\n--- Processing patch {i} ---")
                plot_overlay_and_save(dataset, i, city_name, output_dir)
            
            print(f"\nCompleted processing {num_samples} patches for {city_name}")
            
        except Exception as e:
            print(f"ERROR processing {city_name}: {str(e)}")
            print(f"Skipping to next city...\n")
            continue
    
    print(f"\n{'=' * 80}")
    print(f"All processing complete! Output saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
'''