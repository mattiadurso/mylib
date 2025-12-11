import numpy as np
import torch
from PIL import Image


def read_image_pt(path):
    # Input should be grayscale image with range [0, 1].
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).float()
