import torch
import numpy
import trimesh
from tqdm import tqdm
import numpy as np
from importlib.machinery import SourceFileLoader
import os
epoches = 15
batchsize = 8
dataloadpath = 'dataloader/dataload.py'
dataloadname = 'dataloader'
diffusionpath = 'diffusion/gaussian_diffusion.py'
diffusionname = 'diffusion'
modelpath = 'model/transformer.py'
modelname = 'model'

dataload = SourceFileLoader(dataloadname, dataloadpath).load_module()
diffusionload = SourceFileLoader(diffusionname, diffusionpath).load_module()
modelload = SourceFileLoader(modelname, modelpath).load_module()
PATH = '../../dataset/airplane'


def train(device, dtype, dataloader):
    model = modelload.PointDiffusionTransformer(
        device=device, dtype=dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)
    for epoch in tqdm(range(epoches)):
        model.train()
        pbar = tqdm(total=len(dataloader))
        losses = 0
        for pc in dataloader:
            optimizer.zero_grad()
            pc = pc.to(device)
            t = np.random.rand(pc.shape[0]).astype(np.float32 if device==torch.device('mps') else np.float64)
            
            timestep = torch.randint(1,10,(pc.shape[0],)).to(device)
            # predict = model(pc, timestep).to(device)
            
            diffusion = diffusionload.GaussianDiffusion(t, model).to(device)
            diffusionloss = diffusion.forward(pc, timestep)
            losses+=diffusionloss
            diffusionloss.backward()
            optimizer.step()
            pbar.update(1)
        scheduler.step()
        pbar.close()
        print(f"Epoch: {epoch}, Loss: {losses / len(dataloader)}")
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join('diffusion_model.pth'))

if __name__ == "__main__":
    dataset, dataloader = dataload.Prepare_Dataset(PATH, batch_size=batchsize)
    device = torch.device('cpu')
    dtype = torch.float64
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if torch.backends.mps.is_available():
        dtype = torch.float32
        device = torch.device('mps')
    train(device, dtype, dataloader)
