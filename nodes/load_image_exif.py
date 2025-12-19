"""LoadImageWithExif node for ComfyUI-Sharp.

Loads an image and extracts focal length from EXIF metadata.
"""

import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, ExifTags, TiffTags

try:
    import folder_paths
    import node_helpers
except ImportError:
    folder_paths = None
    node_helpers = None

# Try to import pillow_heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False


def extract_exif(img_pil: Image.Image) -> dict:
    """Extract EXIF information as a dictionary.

    Based on ml-sharp/src/sharp/utils/io.py:extract_exif()
    """
    exif_dict = {}

    try:
        # Get full exif description from get_ifd(0x8769)
        img_exif = img_pil.getexif().get_ifd(0x8769)
        exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}
    except Exception:
        pass

    try:
        # Also get TIFF tags
        tiff_tags = img_pil.getexif()
        tiff_dict = {TiffTags.TAGS_V2[k].name: v for k, v in tiff_tags.items() if k in TiffTags.TAGS_V2}
        exif_dict.update(tiff_dict)
    except Exception:
        pass

    return exif_dict


def extract_focal_length_mm(img_pil: Image.Image, default_mm: float = 30.0) -> float:
    """Extract focal length in mm (35mm equivalent) from EXIF.

    Based on ml-sharp/src/sharp/utils/io.py:load_rgb()

    Args:
        img_pil: PIL Image object
        default_mm: Default focal length if not found in EXIF

    Returns:
        Focal length in mm (35mm film equivalent)
    """
    img_exif = extract_exif(img_pil)

    # Try to get 35mm equivalent focal length first
    f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))

    if f_35mm is None or f_35mm < 1:
        # Fall back to raw focal length
        f_35mm = img_exif.get("FocalLength", None)

        if f_35mm is None:
            print(f"[LoadImageWithExif] No focal length in EXIF - using default {default_mm}mm")
            return default_mm

        # If focal length is very small, it's probably not for 35mm equivalent
        if f_35mm < 10.0:
            print(f"[LoadImageWithExif] Found focal length {f_35mm}mm < 10mm, assuming not 35mm equivalent")
            # This is a crude approximation (assumes typical smartphone sensor crop factor)
            f_35mm *= 8.4

    print(f"[LoadImageWithExif] Extracted focal length: {f_35mm}mm (35mm equivalent)")
    return float(f_35mm)


class LoadImageWithExif:
    """Load an image and extract focal length from EXIF metadata.

    Works like the standard ComfyUI LoadImage node but also outputs
    the focal length extracted from EXIF data (35mm equivalent).
    """

    @classmethod
    def INPUT_TYPES(cls):
        if folder_paths is None:
            return {"required": {"image": ("STRING", {"default": ""})}}

        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "default_focal_mm": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 0.1,
                    "tooltip": "Default focal length (35mm equiv) if not found in EXIF"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT",)
    RETURN_NAMES = ("image", "mask", "focal_length_mm",)
    FUNCTION = "load_image"
    CATEGORY = "SHARP"
    DESCRIPTION = "Load an image and extract focal length from EXIF metadata (35mm equivalent)."

    def load_image(self, image: str, default_focal_mm: float = 30.0):
        """Load image and extract EXIF focal length."""
        if folder_paths is None:
            raise RuntimeError("ComfyUI folder_paths not available")

        image_path = folder_paths.get_annotated_filepath(image)

        # Open image
        img = node_helpers.pillow(Image.open, image_path)

        # Extract focal length before any transforms
        focal_length_mm = extract_focal_length_mm(img, default_focal_mm)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            # Handle EXIF rotation
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            # Skip frames with different sizes
            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            # Convert to tensor
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            # Handle alpha/mask
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

        # Combine frames
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, focal_length_mm)

    @classmethod
    def IS_CHANGED(cls, image, default_focal_mm=30.0):
        if folder_paths is None:
            return image
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, default_focal_mm=30.0):
        if folder_paths is None:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "LoadImageWithExif": LoadImageWithExif,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithExif": "Load Image with EXIF (Focal Length)",
}
