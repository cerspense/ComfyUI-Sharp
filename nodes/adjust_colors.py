"""AdjustGaussianColors node for ComfyUI-Sharp.

Adjusts gamma, brightness, contrast, and saturation of Gaussian splat PLY vertex colors.
Colors are stored as SH degree-0 coefficients — this node converts to RGB, applies
adjustments, and converts back.
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

# SH degree-0 coefficient
C0 = np.sqrt(1.0 / (4.0 * np.pi))


def sh_to_rgb(sh):
    """Convert degree-0 spherical harmonics to RGB [0,1]."""
    return sh * C0 + 0.5


def rgb_to_sh(rgb):
    """Convert RGB [0,1] back to degree-0 spherical harmonics."""
    return (rgb - 0.5) / C0


class AdjustGaussianColors:
    """Adjust the colors of Gaussian splat PLY vertices.

    Applies gamma correction and other color adjustments to work around
    color space mismatches in external viewers/renderers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {
                    "tooltip": "Path to input PLY file",
                    "forceInput": True,
                }),
            },
            "optional": {
                "output_prefix": ("STRING", {
                    "default": "color_adjusted",
                    "tooltip": "Prefix for output PLY filename"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction. <1 brightens (e.g. 0.45 = sRGB encode), >1 darkens (e.g. 2.2 = sRGB decode). 1.0 = no change."
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Additive brightness adjustment. 0 = no change."
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Contrast multiplier around midpoint (0.5). 1.0 = no change."
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Saturation multiplier. 0 = grayscale, 1.0 = no change, >1 = more saturated."
                }),
                "exposure": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Exposure adjustment in stops. 0 = no change, +1 = double brightness, -1 = half."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "adjust"
    CATEGORY = "SHARP"
    OUTPUT_NODE = True
    DESCRIPTION = "Adjust gamma, brightness, contrast, saturation, and exposure of Gaussian splat PLY colors. Use to fix color space mismatches in external viewers."

    def adjust(
        self,
        ply_path: str,
        output_prefix: str = "color_adjusted",
        gamma: float = 1.0,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        exposure: float = 0.0,
    ):
        ply_file = Path(ply_path)
        if not ply_file.exists():
            raise ValueError(f"PLY file not found: {ply_file}")

        log.info(f"Loading PLY: {ply_file}")
        plydata = PlyData.read(str(ply_file))
        vertices = plydata['vertex'].data
        N = len(vertices)

        # Extract SH coefficients and convert to RGB
        sh = np.stack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']], axis=-1)
        rgb = sh_to_rgb(sh)

        log.info(f"Adjusting {N:,} Gaussians (gamma={gamma}, brightness={brightness}, contrast={contrast}, saturation={saturation}, exposure={exposure})")

        # Apply exposure (multiplicative, in stops)
        if exposure != 0.0:
            rgb = rgb * (2.0 ** exposure)

        # Apply gamma correction
        if gamma != 1.0:
            rgb = np.clip(rgb, 0.0, None)
            rgb = np.power(rgb, 1.0 / gamma)

        # Apply contrast (around 0.5 midpoint)
        if contrast != 1.0:
            rgb = (rgb - 0.5) * contrast + 0.5

        # Apply brightness
        if brightness != 0.0:
            rgb = rgb + brightness

        # Apply saturation
        if saturation != 1.0:
            luminance = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
            luminance = luminance[:, np.newaxis]
            rgb = luminance + saturation * (rgb - luminance)

        # Clamp to valid range and convert back to SH
        rgb = np.clip(rgb, 0.0, 1.0)
        sh_out = rgb_to_sh(rgb)

        # Update vertex data (copy to make writable)
        new_vertices = vertices.copy()
        new_vertices['f_dc_0'] = sh_out[:, 0].astype(np.float32)
        new_vertices['f_dc_1'] = sh_out[:, 1].astype(np.float32)
        new_vertices['f_dc_2'] = sh_out[:, 2].astype(np.float32)

        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_filename = f"{output_prefix}_{timestamp}.ply"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        elements = [PlyElement.describe(new_vertices, 'vertex')]
        for element in plydata.elements:
            if element.name != 'vertex':
                elements.append(element)

        PlyData(elements).write(output_path)
        log.info(f"Saved: {output_path}")

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "SharpAdjustGaussianColors": AdjustGaussianColors,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SharpAdjustGaussianColors": "SHARP Adjust Gaussian Colors",
}
