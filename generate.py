import torch
import trimesh
from tqdm import tqdm
import numpy as np
import os
from diffusion.gaussian_diffusion import GaussianDiffusion
from model.transformer import PointDiffusionTransformer
from dataloader.dataload import Prepare_Dataset
from plyfile import PlyData, PlyElement

num_timesteps = 1000
sample_nums = 2048
bs = 1
chosen_category = "airplane"
model_path = "ckpts/uncondition/" + chosen_category + ".ckpt"
output_path = "results/" + chosen_category
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = np.linspace(0.0001, 0.02, num_timesteps, dtype=np.float64)

if __name__ == "__main__":
    noise_predictor = PointDiffusionTransformer(
        device=device, dtype=torch.float32, n_ctx=sample_nums
    ).to(device)
    noise_predictor.load_state_dict(torch.load(model_path))
    noise_predictor.eval()
    diffusion = GaussianDiffusion(betas, noise_predictor)
    with torch.no_grad():
        batch_samples = diffusion.p_sample_loop(
            (bs, 3, sample_nums), device=device, return_process=False
        )
        for i in range(bs):
            sample = batch_samples[i]
            points = sample.cpu().numpy().reshape(3, -1)
            points = np.transpose(points)
            vertex = np.zeros(
                len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
            )
            vertex["x"] = points[:, 0]
            vertex["y"] = points[:, 1]
            vertex["z"] = points[:, 2]

            ply = PlyData([PlyElement.describe(vertex, "vertex")])
            ply.write(output_path + "/" + str(i) + ".ply")
