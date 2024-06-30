import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


def normalizeXZY(point):
    centroid = np.mean(point, axis=0)
    point -= centroid
    distance = np.max(np.sqrt(np.sum(point**2, axis=1)))
    point /= distance
    return point


class ModelNetDataset(Dataset):
    def __init__(self, filepath, num_points=1024, mode="npz", normalize=True):
        self.filepath = filepath
        self.num_points = num_points
        self.normalize = normalize
        self.mode = 1 if mode == "npz" else 0
        self.filesdir = os.listdir(filepath)
        self.filespath = filepath
        self.filesnpz = [i for i in self.filesdir if i.endswith("npz")]
        self.filestxt = [i for i in self.filesdir if i.endswith("txt")]

    def __len__(self):
        return len(self.filesnpz)

    def __getitem__(self, index):
        # class_name, data_path = self.datapath[index]
        # classid = self.classes[class_name]
        # point_set = np.loadtxt(data_path, delimiter=',').astype(np.float32)
        # point_set = point_set[0:self.num_points, :]
        # if self.normalize:
        #     point_set[:, 0:3] = normalizeXZY(point_set[:, 0:3])
        # return point_set, classid
        if self.mode:
            pointcloud_path = self.filesnpz[index]
            pointcloud = np.load(os.path.join(self.filespath, pointcloud_path))["pc"]
            # print(pointcloud)
            pointcloud = np.asarray(pointcloud).astype(np.float32)
        else:
            pointcloud_path = self.filestxt[index]
            pointcloud = np.loadtxt(
                os.path.join(self.filespath, pointcloud_path), delimiter=" "
            ).astype(np.float32)
        if self.normalize:
            pointcloud[:, :3] = normalizeXZY(pointcloud[:, :3])

        pointcloud = np.random.permutation(pointcloud)
        pointcloud = pointcloud[: self.num_points, :]
        return pointcloud

    def dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def Prepare_Dataset(
    filepath,
    num_points=1024,
    mode="npz",
    normalize=True,
    batch_size=32,
    shuffle=True,
    num_workers=0,
):
    dataset = ModelNetDataset(
        filepath, num_points=num_points, mode=mode, normalize=normalize
    )
    dataloader = dataset.dataloader(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataset, dataloader
