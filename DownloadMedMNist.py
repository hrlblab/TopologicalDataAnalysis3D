import os
import numpy as np
import medmnist
from medmnist import INFO

def download_and_save_medmnist(dataset_name, target_dir):
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])


    train_dataset = DataClass(split='train', download=True)
    val_dataset = DataClass(split='val', download=True)
    test_dataset = DataClass(split='test', download=True)


    base_dir = os.path.join(target_dir, dataset_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    def save_data(dataset, split):
        split_dir = os.path.join(base_dir, split)
        for idx, (img, label) in enumerate(zip(dataset.imgs, dataset.labels)):
            label_dir = os.path.join(split_dir, str(label))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            original_filename = dataset.filename[idx] if hasattr(dataset, 'filename') else str(idx)
            img_path = os.path.join(label_dir, f'{original_filename}.npy')
            np.save(img_path, img)

    save_data(train_dataset, 'train')
    save_data(val_dataset, 'val')
    save_data(test_dataset, 'test')

# 示例调用
dataset_names = ['organmnist3d', 'nodulemnist3d', 'fracturemnist3d', 'adrenalmnist3d', 'vesselmnist3d', 'synapsemnist3d']
target_dir = 'data'

for dataset_name in dataset_names:
    download_and_save_medmnist(dataset_name, target_dir)