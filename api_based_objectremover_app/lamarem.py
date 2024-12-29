import os
from typing import Tuple, Union
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch.hub import download_url_to_file
from pathlib import Path

# --- Constants and paths ---
LAMA_MODEL_PATH = Path(__file__).parent / "ckpts"
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Image processing functions ---
def _fit_to_eight(x: int) -> int:
    """
    Adjust the input dimension to be divisible by 8.
    This is often necessary for neural network processing.
    """
    return (x // 8 + 1) * 8 if x % 8 else x

def _pad_image(image: Image.Image, mode: str = "RGB", fill_color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Pad the image to dimensions divisible by 8.
    This ensures compatibility with the neural network's expected input size.
    """
    w, h = image.size
    new_w, new_h = _fit_to_eight(w), _fit_to_eight(h)
    if new_w != w or new_h != h:
        padded_image = Image.new(mode, (new_w, new_h), fill_color)
        padded_image.paste(image, (0, 0))
        return padded_image
    return image

def _image_to_tensor(image: Image.Image, is_mask: bool = False) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch tensor.
    Handles both regular images and mask images.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")
    image = image.convert("RGB") if not is_mask and image.mode != "RGB" else image
    img = np.array(image).astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1)) if img.ndim == 3 else img[None, ...]
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)

def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL Image.
    Handles both single-channel and multi-channel tensors.
    """
    i = 255. * tensor.cpu().numpy().clip(0, 1)
    i = i.astype(np.uint8)
    if i.ndim == 3 and i.shape[0] == 1:
        i = i.squeeze(0)
    elif i.ndim == 3 and i.shape[0] == 3:
        i = np.transpose(i, (1, 2, 0))
    return Image.fromarray(i)

# --- Model loading and inference ---
def _download_model(filename: str, url: str = LAMA_URL, localdir: Path = LAMA_MODEL_PATH) -> Path:
    """
    Download the LaMa model if it doesn't exist locally.
    Returns the path to the downloaded or existing model file.
    """
    model_path = localdir / filename
    if not model_path.exists():
        print(f"Downloading biglama model...")
        localdir.mkdir(parents=True, exist_ok=True)
        download_url_to_file(url, model_path)
    return model_path

class LaMaRemover:
    def __init__(self, model_path: Union[str, Path] = None):
        """
        Initialize the LaMa (Large Mask) inpainting model.
        Loads the model from the specified path or downloads it if not provided.
        """
        self.device = DEVICE
        model_path = model_path or _download_model("big-lama.pt")
        self.model = torch.jit.load(model_path, map_location=self.device).eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, image: Image.Image, mask: Image.Image, blur_radius: int = 3, 
                 mask_threshold: int = 250, invert_mask: bool = False) -> Image.Image:
        """
        Perform inpainting on the input image using the provided mask.
        
        Args:
            image: Input image to be inpainted.
            mask: Mask indicating areas to be inpainted.
            blur_radius: Radius for Gaussian blur applied to the mask.
            mask_threshold: Threshold for binarizing the mask.
            invert_mask: Whether to invert the mask before processing.
        
        Returns:
            Inpainted image as a PIL Image.
        """
        w, h = image.size
        image = _pad_image(image)
        image_tensor = _image_to_tensor(image)

        mask = _pad_image(mask.resize(image.size), mode="L", fill_color=0)
        mask = ImageOps.invert(mask) if not invert_mask else mask
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = mask.point(lambda x: 0 if x > mask_threshold else 255)
        mask_tensor = _image_to_tensor(mask, is_mask=True)

        result_tensor = self.model(image_tensor, mask_tensor)[0]
        result_image = _tensor_to_image(result_tensor)
        return result_image.crop((0, 0, w, h)) if result_image.size != (w, h) else result_image

def main():
    """
    Main function to demonstrate the usage of LaMaRemover.
    Loads an image and a mask, performs inpainting, and saves the result.
    """
    image_path = "image1.png"
    mask_path = "image2.png"

    remover = LaMaRemover()
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    output_image = remover(image, mask)
    output_image.save("output.png")

# --- Example usage ---
if __name__ == "__main__":
    main()