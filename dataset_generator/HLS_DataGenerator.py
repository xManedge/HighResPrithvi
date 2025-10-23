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


class HLSDataset(Dataset):
    
    def __init__(self, image_loc, label_loc, tile = 224, stride = 224, ignore_index = 255, verbose=False):
        
        self.image_loc = image_loc # image location to the blue band B02 
        self.label_loc = label_loc
        self.tile = tile
        self.stride = stride
        self.ignore_index = ignore_index
        self.verbose = verbose
        
        self._setup_aligned_grid()
        
        # Band names for Sentinel-2
        self.band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
        
        # calculate location of all the bands 
        self.all_band_locations = [self.image_loc[:-7] + band_name + self.image_loc[-4:] for band_name in self.band_names]
        
        
        self.windows = self._build_windows()
        
    
    def _setup_aligned_grid(self):
        # load metadata of HLS image
        with rasterio.open(self.image_loc) as src_image: 
            self.hls_transform = src_image.transform
            self.hls_bounds = src_image.bounds
            self.hls_crs = src_image.crs
            self.hls_shape = src_image.shape
    
        # Get label metadata
        with rasterio.open(self.label_loc) as src_labels:
            self.label_transform = src_labels.transform
            original_label_bounds = src_labels.bounds  # Keep original
            self.label_crs = src_labels.crs
            self.label_shape = src_labels.shape
         
        if self.verbose: 
            print(f"[HLS IMAGE]\tmin_x:\t{self.hls_bounds.left}\tmin_y:\t{self.hls_bounds.bottom}\tmax_x:\t{self.hls_bounds.right}\tmax_y:\t{self.hls_bounds.top}")
            print('\n')
            print(f"[LABEL]\tmin_x:\t{original_label_bounds.left}\tmin_y:\t{original_label_bounds.bottom}\tmax_x:\t{original_label_bounds.right}\tmax_y:\t{original_label_bounds.top}")
        
        # Transform label bounds to match HLS coordinate system
        if self.label_crs != self.hls_crs:
            from rasterio.warp import transform_bounds
            self.label_bounds = transform_bounds(
                self.label_crs,
                self.hls_crs,
                *original_label_bounds
            )
            if self.verbose:
                print(f"[LABEL TRANSFORMED]\tmin_x:\t{self.label_bounds[0]}\tmin_y:\t{self.label_bounds[1]}\tmax_x:\t{self.label_bounds[2]}\tmax_y:\t{self.label_bounds[3]}")
        else:
            self.label_bounds = original_label_bounds
            
        # Store original label transform for later use
        self.label_transform = src_labels.transform
        
        # Now calculate intersection - both in HLS coordinate system
        hls_box = box(*self.hls_bounds)
        label_box = box(*self.label_bounds)
        intersection = hls_box.intersection(label_box)
    
        # assert intersection is non empty 
        assert not intersection.is_empty, "Intersection is empty, are you sure your ground truth and your images match"
        
        self.intersection_bounds = intersection.bounds
        
        if self.verbose:        
            print("intersection bounds: ", self.intersection_bounds)
        
        pixel_size = self.hls_transform.a # 30m for this dataset
        
        # Snap intersection bounds to pixel grid
        min_x = np.floor((self.intersection_bounds[0] - self.hls_bounds[0]) / pixel_size) * pixel_size + self.hls_bounds[0]
        min_y = np.floor((self.intersection_bounds[1] - self.hls_bounds[1]) / pixel_size) * pixel_size + self.hls_bounds[1]
        max_x = np.ceil((self.intersection_bounds[2] - self.hls_bounds[0]) / pixel_size) * pixel_size + self.hls_bounds[0]
        max_y = np.ceil((self.intersection_bounds[3] - self.hls_bounds[1]) / pixel_size) * pixel_size + self.hls_bounds[1]
        
        if self.verbose: 
            print(f"min_x:\t{min_x}\tmin_y:\t{min_y}\tmax_x:\t{max_x}\tmax_y:\t{max_y}")
    
        self.aligned_bounds = (min_x, min_y, max_x, max_y)
        
        # Compute aligned dimensions
        self.aligned_width = int((max_x - min_x) / pixel_size)
        self.aligned_height = int((max_y - min_y) / pixel_size)
        
        # Create aligned transform
        self.aligned_transform = transform_from_bounds(
            *self.aligned_bounds, self.aligned_width, self.aligned_height
        )
    
        print(f"Aligned grid: {self.aligned_width}x{self.aligned_height} pixels")
        print(f"Pixel size: {pixel_size}m")


    def _build_windows(self):
        windows = []
        
        
        for r in range(0, self.aligned_height - self.tile + 1, self.stride): 
            for c in range(0, self.aligned_width - self.tile + 1, self.stride): 
                windows.append((r,c))
        
        return windows 
    
    
    def __getitem__(self, idx): 
        row, col = self.windows[idx]
        
        img = self._load_HLS_image(row, col, self.tile)     # (6,tile, tile)
        lbl = self._load_labels_patch(row, col, self.tile)  # (tile, tile)
        
        if self.verbose:
            print(f"[LOADED IMAGE]\tSource:\t{self.image_loc}")
            print(f"\tPatch index:\t{idx}\t(Row: {row}, Col: {col})")
            print(f"\tImage shape:\t{img.shape}\tLabel shape:\t{lbl.shape}")
            print(f"\tImage min:\t{img.min():.4f}\tmax:\t{img.max():.4f}\tmean:\t{img.mean():.4f}") 
            print(f"\tLabel unique values:\t{np.unique(lbl)}")
        

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        
        # insert augmentation code later 
            
        return img, lbl
        
    def _load_HLS_image(self, row, col, size):
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
                
                # Ensure exact size (handle edge cases)
                if band_data.shape != (size, size):
                    # Pad or crop to exact size
                    padded = np.zeros((size, size), dtype=np.float32)
                    h, w = min(band_data.shape[0], size), min(band_data.shape[1], size)
                    padded[:h, :w] = band_data[:h, :w]
                    band_data = padded
                
                bands.append(band_data)
            
        stacked = np.stack(bands)
        
        return stacked

    def _load_labels_patch(self, row, col, size): 
        patch_bounds = (
            self.aligned_bounds[0] + col * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + (row + size) * self.aligned_transform.e,
            self.aligned_bounds[0] + (col + size) * abs(self.aligned_transform.a),
            self.aligned_bounds[3] + row * self.aligned_transform.e
        )
        
        with rasterio.open(self.label_loc) as src: 
            if src.crs != self.hls_crs:
                # Transform patch bounds from HLS CRS to Label CRS
                from rasterio.warp import transform_bounds
                patch_bounds_label_crs = transform_bounds(
                    self.hls_crs,
                    src.crs,
                    *patch_bounds
                )
                
                # Now create window in label's coordinate system
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
                # Same CRS - direct window read
                window = from_bounds(*patch_bounds, transform=src.transform)
                label_patch = src.read(1, window=window)
                
                # Ensure exact size
                if label_patch.shape != (size, size):
                    padded = np.full((size, size), self.ignore_index, dtype=np.uint8)
                    h, w = min(label_patch.shape[0], size), min(label_patch.shape[1], size)
                    padded[:h, :w] = label_patch[:h, :w]
                    label_patch = padded

            # Remap labels
            label_patch = self._remap_labels(label_patch)
        
            return label_patch
            
    def _remap_labels(self, label_patch):
        """
        Remap NLCD labels to simplified 4-class scheme:
        0: Water
        1: Trees/Forest
        2: Built-up/Developed
        3: Grasslands/Rangelands
        255: Ignore
        """
        remapped = np.full_like(label_patch, 255, dtype=np.uint8)
        
        # Water
        remapped[label_patch == 11] = 0
        
        # Trees/Forest (all 40s + woody wetlands + SHRUBLAND)
        remapped[np.isin(label_patch, [41, 42, 43, 52, 90])] = 1
        
        # Built-up/Developed
        remapped[np.isin(label_patch, [22, 23, 24])] = 2
        
        # Grasslands/Rangelands 
        remapped[np.isin(label_patch, [21, 71, 72, 73, 74, 81, 82, 95])] = 3
        
        # Ignore: 12 (ice/snow), 31 (barren), 250 (nodata)
        
        return remapped
        

    def __len__(self): 
        return len(self.windows)
    
    
    def plot_overlay(self, idx, figsize=(15, 5)):
        """
        Plot image and label overlay for a given patch index.
        
        Args:
            idx: Index of the patch to visualize
            figsize: Figure size (width, height)
        """
        # Get the patch
        img, lbl = self[idx]
        
        # Convert from torch tensors to numpy
        img_np = img.numpy()
        lbl_np = lbl.numpy()
        
        # Create RGB composite (B04-R, B03-G, B02-B for true color)
        # Bands are in order: B02, B03, B04, B05, B06, B07
        rgb = np.stack([
            img_np[2],  # B04 - Red
            img_np[1],  # B03 - Green
            img_np[0],  # B02 - Blue
        ], axis=-1)
        
        # Normalize RGB for display (clip to 2-98 percentile for better contrast)
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:,:,i], (2, 98))
            rgb_norm[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2), 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot RGB image
        axes[0].imshow(rgb_norm)
        axes[0].set_title(f'RGB Image (Patch {idx})')
        axes[0].axis('off')
        
        # Plot labels with custom colormap
        colors = ['blue', 'darkgreen', 'red', '#39FF14', 'white']  # Neon green for grasslands
        cmap = ListedColormap(colors)
        
        # Mask ignore values
        lbl_masked = np.ma.masked_equal(lbl_np, 255)
        
        im = axes[1].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, interpolation='nearest')
        axes[1].set_title('Labels')
        axes[1].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Water', 'Trees', 'Built-up', 'Grassland'])
        
        # Plot overlay
        axes[2].imshow(rgb_norm)
        axes[2].imshow(lbl_masked, cmap=cmap, vmin=0, vmax=4, alpha=0.5, interpolation='nearest')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Print statistics
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


def main():
    """
    Test the data generator with sample data.
    """
    # REPLACE THESE WITH YOUR ACTUAL FILE PATHS
    image_loc = "C:/Users/BCC/Desktop/Manoj Work/Prithvi/Codes/Dataset/HLS-2/Chicago/HLS.S30.T16TDM.2025261T164701.v2.0.B02.tif"
    label_loc = "C:/Users/BCC/Desktop/Manoj Work/Prithvi/Codes/Dataset/NLCD/Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"
    
    print("=" * 80)
    print("Initializing Dataset...")
    print("=" * 80)
    
    # Create dataset
    dataset = HLSDataset(
        image_loc=image_loc,
        label_loc=label_loc,
        tile=224,
        stride=224,
        ignore_index=255,
        verbose=True
    )
    
    print(len(dataset))
    print("\n" + "=" * 80)
    print(f"Dataset created successfully!")
    print(f"Total number of patches: {len(dataset)}")
    print("=" * 80)
    
    # Test loading a few patches
    print("\nTesting patch loading...")
    for i in [0, len(dataset)//2, len(dataset)-1]:
        print(f"\nLoading patch {i}...")
        try:
            img, lbl = dataset[i]
            print(f"✓ Patch {i} loaded successfully")
            print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"  Label shape: {lbl.shape}, dtype: {lbl.dtype}")
        except Exception as e:
            print(f"✗ Error loading patch {i}: {e}")
    
    # Visualize some patches
    print("\n" + "=" * 80)
    print("Visualizing patches...")
    print("=" * 80)
    
    # Plot first patch
    i, j, k = np.random.choice(255, 3, replace=False)
    dataset.plot_overlay(i)
    
    # Plot a middle patch
    dataset.plot_overlay(j)
    
    # Plot last patch
    dataset.plot_overlay(k)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()