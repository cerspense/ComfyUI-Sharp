"""LoadSharpModel node for ComfyUI-Sharp."""

import os
import logging

import torch
from huggingface_hub import hf_hub_download

from comfy_api.latest import io

log = logging.getLogger("sharp")

# Try to get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = os.path.join(folder_paths.models_dir, "sharp")
    os.makedirs(MODELS_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("sharp", MODELS_DIR)
except ImportError:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sharp")

SHARP_REPO_ID = "apple/Sharp"
SHARP_FILENAME = "sharp_2572gikvuh.pt"


class LoadSharpModel(io.ComfyNode):
    """Load SHARP model and wrap with ModelPatcher for ComfyUI-native VRAM management.

    The model is built once, cached by ComfyUI's execution cache, and stays in
    VRAM between runs (no repeated GPU transfers).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSharpModel",
            display_name="(Down)Load SHARP Model",
            category="SHARP",
            description="Load the SHARP model for monocular 3D Gaussian Splatting prediction.",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                               default="auto", optional=True,
                               tooltip="Model precision. auto: best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
            ],
            outputs=[
                io.Custom("SHARP_MODEL").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, precision: str = "auto"):
        """Build model, load weights, wrap with ModelPatcher."""
        import comfy.model_management
        import comfy.model_patcher
        import comfy.ops
        import comfy.utils
        from .sharp import PredictorParams, create_predictor

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        # Resolve dtype
        if precision == "auto":
            if comfy.model_management.should_use_bf16(load_device):
                dtype = torch.bfloat16
            elif comfy.model_management.should_use_fp16(load_device):
                dtype = torch.float16
            else:
                dtype = torch.float32
        elif precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Select optimal operations class (enables fp8, nvfp4, CublasOps, etc.)
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype)

        # Download checkpoint if needed
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=SHARP_REPO_ID,
            filename=SHARP_FILENAME,
            local_dir=MODELS_DIR,
        )

        # Load state dict
        log.info(f"Loading checkpoint from {model_path}")
        state_dict = comfy.utils.load_torch_file(model_path)

        # Build model on meta device (zero memory allocation) then load weights
        # directly with assign=True, avoiding 2x RAM peak from CPU construction.
        log.info("Initializing model on meta device...")
        with torch.device("meta"):
            predictor = create_predictor(
                PredictorParams(),
                dtype=dtype,
                device=None,
                operations=operations,
            )

        # Load weights with assign=True — replaces meta Parameters with real tensors
        # without ever allocating the full model on CPU first.
        # strict=False because registered buffers (e.g. normalization stats) may not
        # be in the checkpoint and will be fixed up below.
        predictor.load_state_dict(state_dict, strict=False, assign=True)

        # Fix any leftover meta-device buffers (e.g. register_buffer constants not
        # present in the checkpoint) by materializing them as real zero tensors.
        for name, buf in list(predictor.named_buffers()):
            if buf.device.type == "meta":
                parts = name.split(".")
                parent = predictor
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                parent._buffers[parts[-1]] = torch.zeros_like(buf, device="cpu")
        predictor.eval()
        if comfy.model_management.force_channels_last():
            predictor.to(memory_format=torch.channels_last)
        comfy.model_management.archive_model_dtypes(predictor)
        log.info(f"Model ready ({dtype})")

        # Wrap with ModelPatcher — ComfyUI manages VRAM from here
        patcher = comfy.model_patcher.ModelPatcher(
            predictor,
            load_device=load_device,
            offload_device=offload_device,
        )

        return io.NodeOutput(patcher)


NODE_CLASS_MAPPINGS = {
    "LoadSharpModel": LoadSharpModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSharpModel": "(Down)Load SHARP Model",
}
