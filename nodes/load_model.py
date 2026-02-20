"""LoadSharpModel node for ComfyUI-Sharp."""

import os
import logging

import torch
from huggingface_hub import hf_hub_download
import comfy.model_management
import comfy.model_patcher

log = logging.getLogger("sharp")

# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"


def _build_sharp_model(model_path, dtype):
    """Build and load the SHARP predictor model.

    Called lazily by inference nodes on first use.
    Returns a loaded nn.Module on CPU with the given dtype.
    """
    from .sharp.models import PredictorParams, create_predictor
    import comfy.utils

    log.info(f"Loading checkpoint from {model_path}")
    state_dict = comfy.utils.load_torch_file(model_path)

    log.info("Initializing model...")
    with torch.device("meta"):
        predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict, assign=True)
    predictor.eval()
    predictor.to(dtype=dtype)  # set dtype, stay on CPU
    log.info(f"Model ready ({dtype})")
    return predictor


class LoadSharpModel:
    """Configure the SHARP model. Downloads checkpoint if needed.

    Model weights are loaded lazily by inference nodes on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .pt checkpoint. Leave empty to auto-download from Hugging Face."
                }),
            }
        }

    RETURN_TYPES = ("SHARP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SHARP"
    DESCRIPTION = "Configure the SHARP model for monocular 3D Gaussian Splatting prediction."

    def load_model(self, precision: str = "auto", checkpoint_path: str = ""):
        """Resolve config and download model if needed. No weight loading."""
        device = comfy.model_management.get_torch_device()

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Resolve / download checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        else:
            os.makedirs(MODELS_DIR, exist_ok=True)
            model_path = hf_hub_download(
                repo_id=SHARP_REPO_ID,
                filename=SHARP_FILENAME,
                local_dir=MODELS_DIR,
            )

        return ({"model_path": model_path, "dtype": dtype},)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "Load SHARP Model",
}
