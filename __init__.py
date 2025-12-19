"""ComfyUI-Sharp: SHARP monocular 3D Gaussian Splatting for ComfyUI.

SHARP (Sharp Monocular View Synthesis) takes a single image and produces
3D Gaussian Splatting representations in under 1 second.

Based on Apple's SHARP model: https://arxiv.org/abs/2512.10685
"""

import os
import sys

# Add this directory to sys.path so the vendored 'sharp' package can be imported
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
