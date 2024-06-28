# Diffusion
A simple diffusion-based 3d generation model

## 简介

model文件夹里是预测噪声的模型noise predictor，其功能为，输入一个尺寸为[B,C,N]的点云x_t和尺寸为[B,]的时间步t，其中B代表batch size, C代表通道数（如只有xyz坐标时C=3，如果是xyzrgb的话C=6），N为采样点的数量，时间步t为一个[B,]的张量，表示每个点云所取的不同时间步。模型预测一个形状为[B,C,N]的噪声。

gaussian_diffusion.py里实现了扩散模型，其训练过程为，给定x_0和随机选择的t，随机采样正态噪声episilon，利用q分布（即原论文中定义的逐步加噪声的过程）生成一个加噪后的x_t，而后noise predictor以x_t和t为输入，预测一个噪声，要求预测出的噪声与episilon尽可能接近。
