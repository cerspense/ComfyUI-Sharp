"""ComfyUI-Sharp: SHARP monocular 3D Gaussian Splatting for ComfyUI.

SHARP (Sharp Monocular View Synthesis) takes a single image and produces
3D Gaussian Splatting representations in under 1 second.

Based on Apple's SHARP model: https://arxiv.org/abs/2512.10685
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
