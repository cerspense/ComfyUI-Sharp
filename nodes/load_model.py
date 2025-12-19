"""LoadSharpModel node for ComfyUI-Sharp."""

import os

import torch

# Model cache
_MODEL_CACHE = {}

# Default model URL (Hugging Face mirror)
DEFAULT_MODEL_URL = "https://huggingface.co/apple/Sharp/resolve/main/sharp_2572gikvuh.pt"


class LoadSharpModel:
    """Load and cache the SHARP model for 3D Gaussian prediction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run inference on. 'auto' selects best available."
                }),
            },
            "optional": {
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
    DESCRIPTION = "Load the SHARP model for monocular 3D Gaussian Splatting prediction."

    def load_model(self, device: str, checkpoint_path: str = ""):
        """Load and cache the SHARP model."""
        from sharp.models import PredictorParams, create_predictor

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "mps") and torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Create cache key
        cache_key = f"{checkpoint_path or 'default'}_{device}"

        if cache_key in _MODEL_CACHE:
            return (_MODEL_CACHE[cache_key],)

        # Load or download checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
        else:
            print(f"[SHARP] Downloading model from Hugging Face...")
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

        # Create predictor
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(device)

        model_dict = {
            "predictor": predictor,
            "device": device,
        }

        _MODEL_CACHE[cache_key] = model_dict

        return (model_dict,)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "Load SHARP Model",
}
