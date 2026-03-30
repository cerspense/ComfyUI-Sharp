"""FilterGaussians node for ComfyUI-Sharp.

Cleans up Gaussian splat PLY files by removing low-quality points using
opacity, depth, scale, and spatial outlier filtering.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

log = logging.getLogger("sharp")

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


class FilterGaussians:
    """Filter and clean up Gaussian splat PLY files.

    Removes low-quality Gaussians using multiple filtering strategies:
    - Opacity: remove nearly-transparent splats
    - Depth percentile: remove distant background splats
    - Scale: remove abnormally large or tiny splats
    - Spatial outliers: remove splats far from the point cloud center
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "tooltip": "Path to input PLY file (from SHARP Predict or Merge Gaussians)",
                    "forceInput": True,
                }),
            },
            "optional": {
                "output_prefix": ("STRING", {
                    "default": "filtered",
                    "tooltip": "Prefix for output PLY filename"
                }),
                "opacity_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Remove Gaussians with opacity below this (PLY stores logits, converted automatically). 0 = disabled."
                }),
                "depth_prune_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Keep only the closest X% of Gaussians by depth. 0.9 = keep 90%, 1.0 = disabled."
                }),
                "max_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Remove Gaussians with any scale axis above this value (in world units). 0 = disabled."
                }),
                "scale_outlier_sigma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Remove Gaussians with scale > mean + N*std. 3.0 is conservative, 2.0 is aggressive. 0 = disabled."
                }),
                "spatial_outlier_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Remove the X% of Gaussians furthest from the point cloud centroid. 0 = disabled."
                }),
                "min_scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Remove Gaussians with all scale axes below this value (dust/noise). 0 = disabled."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT",)
    RETURN_NAMES = ("ply_path", "num_kept", "num_removed",)
    FUNCTION = "filter"
    CATEGORY = "SHARP"
    OUTPUT_NODE = True
    DESCRIPTION = "Clean up Gaussian splat PLY by filtering out low-quality points (low opacity, outlier scale, distant depth, spatial outliers)."

    def filter(
        self,
        ply_path: str,
        output_prefix: str = "filtered",
        opacity_threshold: float = 0.05,
        depth_prune_percent: float = 1.0,
        max_scale: float = 0.0,
        scale_outlier_sigma: float = 0.0,
        spatial_outlier_percent: float = 0.0,
        min_scale: float = 0.0,
    ):
        ply_file = Path(ply_path)
        if not ply_file.exists():
            raise ValueError(f"PLY file not found: {ply_file}")

        # Handle folder input (from batch predict) - filter each file
        if ply_file.is_dir():
            return self._filter_folder(
                ply_file, output_prefix, opacity_threshold,
                depth_prune_percent, max_scale, scale_outlier_sigma,
                spatial_outlier_percent, min_scale,
            )

        log.info(f"Loading Gaussians from: {ply_file}")
        plydata = PlyData.read(str(ply_file))
        vertices = plydata['vertex'].data
        N = len(vertices)
        log.info(f"Loaded {N:,} Gaussians")

        # Extract arrays for filtering
        xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
        opacity_logits = vertices['opacity']
        scales = np.stack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']], axis=1)

        # Scales are stored in log-space in the PLY
        actual_scales = np.exp(scales)

        mask = np.ones(N, dtype=bool)

        # 1. Opacity filtering
        if opacity_threshold > 0:
            actual_opacity = 1.0 / (1.0 + np.exp(-opacity_logits))
            opacity_mask = actual_opacity >= opacity_threshold
            removed = (~opacity_mask & mask).sum()
            mask &= opacity_mask
            log.info(f"Opacity filter (>={opacity_threshold:.2f}): removed {removed:,}, remaining {mask.sum():,}")

        # 2. Depth percentile pruning
        if depth_prune_percent < 1.0:
            valid_depths = xyz[mask, 2]
            if len(valid_depths) > 0:
                threshold = np.percentile(valid_depths, depth_prune_percent * 100)
                depth_mask = xyz[:, 2] <= threshold
                removed = (~depth_mask & mask).sum()
                mask &= depth_mask
                log.info(f"Depth pruning (keep {depth_prune_percent*100:.0f}%): removed {removed:,}, remaining {mask.sum():,}")

        # 3. Max scale filtering (hard cutoff)
        if max_scale > 0:
            scale_max_per_gaussian = actual_scales.max(axis=1)
            scale_mask = scale_max_per_gaussian <= max_scale
            removed = (~scale_mask & mask).sum()
            mask &= scale_mask
            log.info(f"Max scale filter (<={max_scale:.2f}): removed {removed:,}, remaining {mask.sum():,}")

        # 4. Scale outlier filtering (statistical)
        if scale_outlier_sigma > 0:
            # Use mean of the 3 scale axes per Gaussian
            scale_mean = actual_scales[mask].mean(axis=1)
            mu = scale_mean.mean()
            sigma = scale_mean.std()
            threshold = mu + scale_outlier_sigma * sigma

            scale_per = actual_scales.mean(axis=1)
            sigma_mask = scale_per <= threshold
            removed = (~sigma_mask & mask).sum()
            mask &= sigma_mask
            log.info(f"Scale outlier filter ({scale_outlier_sigma:.1f}σ, threshold={threshold:.4f}): removed {removed:,}, remaining {mask.sum():,}")

        # 5. Min scale filtering (remove dust/noise)
        if min_scale > 0:
            scale_max_per_gaussian = actual_scales.max(axis=1)
            dust_mask = scale_max_per_gaussian >= min_scale
            removed = (~dust_mask & mask).sum()
            mask &= dust_mask
            log.info(f"Min scale filter (>={min_scale:.4f}): removed {removed:,}, remaining {mask.sum():,}")

        # 6. Spatial outlier removal (distance from centroid)
        if spatial_outlier_percent > 0:
            valid_positions = xyz[mask]
            centroid = valid_positions.mean(axis=0)
            distances = np.linalg.norm(xyz - centroid, axis=1)

            # Only consider currently valid points for the percentile
            valid_distances = distances[mask]
            keep_count = int(len(valid_distances) * (100 - spatial_outlier_percent) / 100)
            if keep_count > 0:
                threshold = np.sort(valid_distances)[min(keep_count, len(valid_distances) - 1)]
                spatial_mask = distances <= threshold
                removed = (~spatial_mask & mask).sum()
                mask &= spatial_mask
                log.info(f"Spatial outlier filter (remove {spatial_outlier_percent:.1f}%): removed {removed:,}, remaining {mask.sum():,}")

        # Apply filter
        filtered_vertices = vertices[mask]
        num_kept = len(filtered_vertices)
        num_removed = N - num_kept

        if num_kept == 0:
            raise ValueError("All Gaussians were filtered out — loosen the filter settings.")

        log.info(f"Final: kept {num_kept:,} / {N:,} ({num_kept/N*100:.1f}%), removed {num_removed:,}")

        # Save filtered PLY
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Preserve extra elements (metadata) from original PLY
        elements = [PlyElement.describe(filtered_vertices, 'vertex')]
        for element in plydata.elements:
            if element.name != 'vertex':
                elements.append(element)

        PlyData(elements).write(output_path)
        log.info(f"Saved filtered PLY: {output_path}")

        return (output_path, num_kept, num_removed,)

    def _filter_folder(
        self, folder, output_prefix, opacity_threshold,
        depth_prune_percent, max_scale, scale_outlier_sigma,
        spatial_outlier_percent, min_scale,
    ):
        """Filter all PLY files in a folder (batch mode)."""
        ply_files = sorted(folder.glob("*.ply"))
        if not ply_files:
            raise ValueError(f"No PLY files found in: {folder}")

        output_folder = os.path.join(OUTPUT_DIR, f"{output_prefix}_{int(time.time() * 1000)}")
        os.makedirs(output_folder, exist_ok=True)

        total_kept = 0
        total_removed = 0

        for ply_file in ply_files:
            result_path, kept, removed = self.filter(
                str(ply_file),
                output_prefix=os.path.join(output_folder, ply_file.stem),
                opacity_threshold=opacity_threshold,
                depth_prune_percent=depth_prune_percent,
                max_scale=max_scale,
                scale_outlier_sigma=scale_outlier_sigma,
                spatial_outlier_percent=spatial_outlier_percent,
                min_scale=min_scale,
            )
            total_kept += kept
            total_removed += removed

        log.info(f"Batch filter done: {len(ply_files)} files, kept {total_kept:,}, removed {total_removed:,}")
        return (output_folder, total_kept, total_removed,)


NODE_CLASS_MAPPINGS = {
    "SharpFilterGaussians": FilterGaussians,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpFilterGaussians": "SHARP Filter Gaussians",
}
