import torch
import trimesh
from tqdm import tqdm
import numpy as np
import os
from diffusion.gaussian_diffusion import GaussianDiffusion
from model.transformer import PointDiffusionTransformer
from dataloader.dataload import Prepare_Dataset
from plyfile import PlyData, PlyElement
import argparse
from config import CONFIG

if __name__ == "__main__":

    chosen_category = CONFIG["category"]
    num_timesteps = CONFIG["num_timesteps"]
    sample_nums = CONFIG["gen_sample"]
    bs = CONFIG["gen_bs"]

    model_path = "ckpts/uncondition/" + chosen_category + ".ckpt"
    output_path = "results/" + chosen_category
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = np.linspace(
        CONFIG["beta_min"], CONFIG["beta_max"], num_timesteps, dtype=np.float64
    )
    noise_predictor = PointDiffusionTransformer(
        device=device, dtype=torch.float32, n_ctx=sample_nums
    ).to(device)
    noise_predictor.load_state_dict(torch.load(model_path))
    noise_predictor.eval()
    diffusion = GaussianDiffusion(betas, noise_predictor)
    with torch.no_grad():
        batch_samples = diffusion.p_sample_loop(
            (bs, 3, sample_nums), device=device, return_process=True
        )
        for t, samples in enumerate(batch_samples):

            for i in range(bs):
                sample = samples[i]
                points = sample.cpu().numpy().reshape(3, -1)
                points = np.transpose(points)
                vertex = np.zeros(
                    len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
                )
                vertex["x"] = points[:, 0]
                vertex["y"] = points[:, 1]
                vertex["z"] = points[:, 2]

                ply = PlyData([PlyElement.describe(vertex, "vertex")])

                step = num_timesteps - t

                ply.write(
                    output_path
                    + "/i_"
                    + str(i)
                    + "_t_"
                    + str(num_timesteps - t)
                    + ".ply"
                )
