import open3d as o3d
import matplotlib.pyplot as plt
import imageio
import numpy as np
import argparse
from tqdm import tqdm


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str, default="airplane")
parser.add_argument("--batch_num", type=int, default=0)
args = parser.parse_args()
chosen_category = args.category
batch_num = args.batch_num

file_path_prefix = "results/" + chosen_category + "/i_" + str(batch_num) + "_t_"
file_path_postfix = ".ply"
t_list = [i for i in range(0, 100)]

save_path = "results/gif/" + chosen_category + "_i_" + str(batch_num) + ".gif"

ply_files = [file_path_prefix + str(i) + file_path_postfix for i in reversed(t_list)]
images = []


for ply_file in tqdm(ply_files):
    # 加载点云
    pcd = o3d.io.read_point_cloud(ply_file)

    # 将点云转换为numpy数组
    points = np.asarray(pcd.points)

    # 创建图像，设置画布尺寸
    fig = plt.figure(figsize=(8, 8))  # 可以调整尺寸
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=1)

    # 设置相机视角，从斜上方拍摄
    ax.view_init(
        elev=20, azim=45
    )  # elev=45 表示从45度仰角拍摄，azim=45 表示从45度方位角拍摄

    # 设置轴的比例，使各轴的比例一致
    set_axes_equal(ax)

    # 设置相机位置
    ax.set_proj_type("persp")  # 使用透视投影

    # 保存当前帧
    plt.axis("off")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(
        height, width, 4
    )
    image = image[:, :, :3]  # 只保留RGB通道，去除Alpha通道
    images.append(image)
    plt.close()


# 保存为GIF
imageio.mimsave(save_path, images, fps=5)
