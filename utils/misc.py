from PIL import Image
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math

def torch_to_pil(images):
    images = images.detach().cpu().float()
    if images.ndim == 3:
        images = images[None, ...]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) * 0.5
    images = (images * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

