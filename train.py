from .diffusion import gaussian_diffusion
from .model import transformer

import torch
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"


noise_predictor = transformer.PointDiffusionTransformer(
    device=device, dtype=torch.float32
)
