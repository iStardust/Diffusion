# Diffusion
A simple diffusion-based 3d generation model

## 简介

model文件夹里是预测噪声的模型noise predictor，其功能为，输入一个尺寸为[B,C,N]的点云x_t和尺寸为[B,]的时间步t，其中B代表batch size, C代表通道数（如只有xyz坐标时C=3，如果是xyzrgb的话C=6），N为采样点的数量，时间步t为一个[B,]的张量，表示每个点云所取的不同时间步。模型预测一个形状为[B,C,N]的噪声。

gaussian_diffusion.py里实现了扩散模型，其训练过程为，给定x_0和随机选择的t，随机采样正态噪声episilon，利用q分布（即原论文中定义的逐步加噪声的过程）生成一个加噪后的x_t，而后noise predictor以x_t和t为输入，预测一个噪声，要求预测出的噪声与episilon尽可能接近。


noise predictor训练完成后，利用p_sample，从x_T开始逐渐采样生成x_0.

## 文件结构
```
dataloader/
    dataset/
        dataset/
            airplane/
            car/
            chair/
            gun/
            table/
        dataload.py
        
```

## 训练

运行`python train.py`进行训练。修改`chosen_category`参数改变类别（airplane,car,chair,rifle,table），训练过程中会不断取损失最小的模型保存到文件夹`ckpts/uncondition/`下。

## 生成

运行`python generate.py`进行生成。给定`chosen_category`，从文件夹`ckpts/uncondition/`里读取已经训练好的模型，进行生成，并将结果保存在`results/chosen_category`下（例如`chosen_category="airplane"`那么就是保存在`results/airplane`下）。`bs`参数代表一次生成几个模型。

## TODO

1. 现在是只支持用点云数据训练并生成点云，考虑增加SDF，三平面的生成模型。
2. 跑实验，把五类的模型和生成结果都跑出来试一试。
3. 把`train.py`和`generate.py`写的漂亮一些，可以考虑把参数统一写进`config.py`文件里？或者以命令行参数的形式设置。