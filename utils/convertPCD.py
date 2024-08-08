import numpy as np
from scipy.ndimage import gaussian_filter


def convert3DtoPCD(img, superpixel, gaussian_sigma):
    # 如果输入的图像是 4D，将其压缩为 3D
    if len(img.shape) == 4:
        img = np.squeeze(img)

    depth, height, width = img.shape[:3]
    if len(img.shape) > 3:
        n_col_chan = img.shape[3]
    else:
        n_col_chan = 1

    n_pixels = np.prod(img.shape[:3])
    list_cubes = [8, 27, 64, 125, 216, 343, 512]
    list_pos = [(1, 0, 0), (1, 1, 1), (2, 1, 1), (2, 2, 2), (3, 2, 2), (3, 3, 3), (4, 3, 3)]
    list_steps = [2, 3, 4, 5, 6, 7, 8]

    step_idx = np.argmin([np.abs(n_pixels / s - superpixel) for s in list_cubes])
    step = list_steps[step_idx]

    # 计算填充的边界
    dh = int(np.ceil(height / step)) * step - height
    dw = int(np.ceil(width / step)) * step - width
    dd = int(np.ceil(depth / step)) * step - depth

    if dh > 0:
        dhI = img[:, -dh:, :]
        img = np.concatenate((img, dhI[:, ::-1, :]), axis=1)
    if dw > 0:
        dwI = img[:, :, -dw:]
        img = np.concatenate((img, dwI[:, :, ::-1]), axis=2)
    if dd > 0:
        ddI = img[-dd:, :, :]
        img = np.concatenate((img, ddI[::-1, :, :]), axis=0)

    grid_z, grid_y, grid_x = np.mgrid[:img.shape[0], :img.shape[1], :img.shape[2]]
    mean_x = grid_x[list_pos[step_idx][0]::step, list_pos[step_idx][1]::step, list_pos[step_idx][2]::step]
    mean_y = grid_y[list_pos[step_idx][0]::step, list_pos[step_idx][1]::step, list_pos[step_idx][2]::step]
    mean_z = grid_z[list_pos[step_idx][0]::step, list_pos[step_idx][1]::step, list_pos[step_idx][2]::step]

    if gaussian_sigma > 0:
        img_blurred = gaussian_filter(img, sigma=gaussian_sigma * np.floor(0.5 * step) / 4)
    else:
        img_blurred = img

    # 将图像转换为点云4D数据
    # 使用 mean_z, mean_y, mean_x 获取图像中的点
    if n_col_chan == 1:
        intensity = img_blurred[mean_z, mean_y, mean_x]
    else:
        intensity = img_blurred[mean_z, mean_y, mean_x].mean(axis=-1)

    point_cloud_data = np.stack((mean_z, mean_y, mean_x, intensity * 255), axis=-1).reshape((-1, 4))

    return point_cloud_data

