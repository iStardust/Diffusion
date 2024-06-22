import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Sequence, Tuple


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    Args:
        betas (Sequence[float]): a 1-D sequence of betas for each diffusion timestep,
            starting at 0 and going to T-1.
        noise_predictor (nn.Module): a noise predictor, which should take as input a
          B*N*C point clouds and a time step t, and regress a B*N*C noise.
    """

    def __init__(self, betas: Sequence[float], noise_predictor: nn.Module):

        super().__init__()
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.noise_predictor = noise_predictor

    def forward(self, x_start: torch.Tensor, t):
        """
        Given x_0 and time step t, returns the mse loss between the
        predicted noise and the ground truth noise.
        """
        gt_noise = torch.randn_like(x_start, device=x_start.device)
        x_t = (
            self.sqrt_alphas_cumprod[t] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t] * gt_noise
        )
        pred_noise = self.noise_predictor(x_t, t)
        loss = F.mse_loss(pred_noise, gt_noise)
        return loss

    def sampling_step_t(self, x_t, t, pred_noise, z):
        """
        returns x_{t-1}
        """
        return np.sqrt(self.betas[t]) * z + (1.0 / np.sqrt(self.alphas[t])) * (
            x_t
            - ((1.0 - self.alphas[t]) / self.sqrt_one_minus_alphas_cumprod[t])
            * pred_noise
        )

    def p_sampling(self, shape, device):
        x_t = torch.randn(*shape, device=device)
        from tqdm import tqdm

        for i in tqdm(reversed(range(0, self.num_timesteps))):
            with torch.no_grad():
                pred_noise = self.noise_predictor(x_t, i)
                z = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
                x_next = self.sampling_step_t(x_t, i, pred_noise, z)
                x_t = x_next

        return x_t


if __name__ == "__main__":
    """
    testing code
    """

    class testNoisePredictor(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc = nn.Linear(3, 3)

        def forward(self, x_t, t):
            return self.fc(x_t)

    noise_predictor = testNoisePredictor()
    diffusion = GaussianDiffusion(
        betas=[0.001, 0.01, 0.1], noise_predictor=noise_predictor
    )
    B = 32
    C = 3
    N = 1024
    x_start = torch.randn(B, N, C, device="cpu")
    T = 3

    num = 10
    while num:
        num -= 1
        t = np.random.randint(0, T)
        loss = diffusion(x_start, t)
        optimizer = torch.optim.SGD(noise_predictor.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss: {}".format(loss))

    x_0 = diffusion.p_sampling((B, N, C), "cpu")
    print(x_0)
