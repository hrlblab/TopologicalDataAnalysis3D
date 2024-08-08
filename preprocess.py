import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.convertPCD import convert3DtoPCD
from utils.getPersistenceImage import persistenceImage
from utils.ImageLoader import PersistenceImageDataset


def convertDataset2PCD(input_dir, output_dir, superpixel, gaussian_sigma):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                img = np.load(file_path)
                point_cloud_data = convert3DtoPCD(img, superpixel, gaussian_sigma)

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_file = os.path.join(output_subdir, os.path.splitext(file)[0] + '_pcd.npy')
                np.save(output_file, point_cloud_data)
                print(f'Saved point cloud data to {output_file}')

def convertPointCloudDataset2PI(input_dir, output_dir, rv_eplision, resolution, bandwidth_eplision, homology_group):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                point_cloud_data = np.load(file_path)
                persistence_image = persistenceImage(point_cloud_data, rv_eplision=rv_eplision, resolution=resolution, bandwidth_eplision=bandwidth_eplision, homology_group=homology_group)

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_file_png = os.path.join(output_subdir, os.path.splitext(file)[0] + '_pi.png')
                persistence_image = persistence_image.reshape(resolution, resolution)
                persistence_image = (persistence_image * 255).astype(np.uint8)
                img = Image.fromarray(persistence_image)
                img.save(output_file_png)
                print(f'Saved persistence image to {output_file_png}')

if __name__ == '__main__':
    superpixels = 600
    gaussian_sigma = -1

    rv_eplision = 30
    resolution = 100
    bandwidth_eplision = 0.95
    homology_group = 2

    dataset_dir = 'data/organmnist3d/train'
    point_cloud_dir = 'data/organmnist3d_pcd/train'
    persistence_img_dir = 'data/organmnist3d_pi/train'

    convertDataset2PCD(input_dir=dataset_dir, output_dir=point_cloud_dir, superpixel=superpixels,
                       gaussian_sigma=gaussian_sigma)
    convertPointCloudDataset2PI(input_dir=point_cloud_dir, output_dir=persistence_img_dir, rv_eplision=rv_eplision,
                                resolution=resolution, bandwidth_eplision=bandwidth_eplision,
                                homology_group=homology_group)

    dataset_dir = 'data/organmnist3d/test'
    point_cloud_dir = 'data/organmnist3d_pcd/test'
    persistence_img_dir = 'data/organmnist3d_pi/test'

    convertDataset2PCD(input_dir=dataset_dir, output_dir=point_cloud_dir, superpixel=superpixels, gaussian_sigma=gaussian_sigma)
    convertPointCloudDataset2PI(input_dir=point_cloud_dir, output_dir=persistence_img_dir, rv_eplision=rv_eplision, resolution=resolution, bandwidth_eplision=bandwidth_eplision, homology_group=homology_group)


