import torch
from tqdm import tqdm
import numpy as np
from diffusion.gaussian_diffusion import GaussianDiffusion
from model.transformer import PointDiffusionTransformer
from dataloader.dataload import Prepare_Dataset
import argparse
from config import CONFIG

if __name__ == "__main__":
    print("Using config {}".format(CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = CONFIG["bs"]
    lr = CONFIG["lr"]
    epochs = CONFIG["epochs"]
    num_timesteps = CONFIG["num_timesteps"]
    chosen_category = CONFIG["category"]
    file_path = "dataloader/dataset/dataset/" + chosen_category
    sample_points = 512
    dataset, dataloader = Prepare_Dataset(
        filepath=file_path,
        batch_size=batch_size,
        num_workers=CONFIG["num_workers"],
        num_points=sample_points,
    )
    betas = np.linspace(
        CONFIG["beta_min"], CONFIG["beta_max"], num_timesteps, dtype=np.float64
    )
    model_path = "ckpts/uncondition/" + chosen_category + ".ckpt"

    noise_predictor = PointDiffusionTransformer(
        device=device, dtype=torch.float32, n_ctx=sample_points
    ).to(device)
    diffusion = GaussianDiffusion(betas, noise_predictor).to(device)
    optimizer = torch.optim.Adam(noise_predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        print("Epoch:{}/{} ".format(epoch + 1, epochs))
        running_loss = []
        for pc in tqdm(dataloader):
            pc = pc.to(device)
            pc = pc.permute(0, 2, 1)
            optimizer.zero_grad()
            batch_t = torch.randint(0, num_timesteps, (pc.shape[0],)).to(device)

            loss = diffusion(pc, batch_t)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        scheduler.step()
        loss = np.mean(running_loss)
        print("Loss: ", loss)
        print("Model saved at {}".format(model_path))
        torch.save(noise_predictor.state_dict(), model_path)
