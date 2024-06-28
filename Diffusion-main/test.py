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
modelpath = 'diffusion_model.pth'

checkpoint = torch.load(modelpath)
device = torch.device('cpu')
dtype = torch.float64
if torch.cuda.is_available():
    device = torch.device('cuda')
if torch.backends.mps.is_available():
    dtype = torch.float32
    device = torch.device('mps')
model = modelload.PointDiffusionTransformer(
        device=device, dtype=dtype).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

