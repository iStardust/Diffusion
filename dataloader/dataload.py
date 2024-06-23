import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

PATH = '../dataset/airplane'


def normalizeXZY(point):
    centroid = np.mean(point, axis=0)
    point -= centroid
    distance = np.max(np.sqrt(np.sum(point ** 2, axis=1)))
    point /= distance
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, filepath, num_points=1024, mode='train', normalize=True):
        self.filepath = filepath
        self.num_points = num_points
        self.normalize = normalize

        self.filesdir = os.listdir(os.path.join(PATH))
        self.files = [i for i in self.filesdir if i.endswith('npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # class_name, data_path = self.datapath[index]
        # classid = self.classes[class_name]
        # point_set = np.loadtxt(data_path, delimiter=',').astype(np.float32)
        # point_set = point_set[0:self.num_points, :]
        # if self.normalize:
        #     point_set[:, 0:3] = normalizeXZY(point_set[:, 0:3])
        # return point_set, classid
        pointcloud_path = self.files[index]
        pointcloud = np.load(os.path.join(PATH, pointcloud_path))['pc']
        return pointcloud

    def dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def Prepare_Dataset(filepath, num_points=1024, mode='train', normalize=True, batch_size=32, shuffle=True, num_workers=0):
    dataset = ModelNetDataLoader(
        filepath, num_points=num_points, mode=mode, normalize=normalize)
    dataloader = dataset.dataloader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, dataloader


dataset, dataloader = Prepare_Dataset(PATH)
