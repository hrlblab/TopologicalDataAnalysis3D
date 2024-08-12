import torch
from torchvision import transforms
import torch.nn.functional as F
import os

import numpy as np
import os
import shutil
from pathlib import Path

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import json

def datafolder_rename(train_dir, test_dir):
    class_list = sorted(os.listdir(train_dir))
    new_class = list(map(str, range(len(class_list))))

    name_mapping = {}
    name_mapping['map'] = {k: v for k, v in zip(class_list, new_class)}
    name_mapping['inv'] = {k: v for k, v in zip(new_class, class_list)}

    json.dump(name_mapping, open(os.path.join(train_dir, '..', 'name_mapping.json'), 'w+'))

    os.rename(train_dir, train_dir+'_raw')
    os.rename(test_dir, test_dir+'_raw')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    for i in range(len(class_list)):
        shutil.copytree(os.path.join(train_dir+'_raw', class_list[i]), os.path.join(train_dir, new_class[i]))
        shutil.copytree(os.path.join(test_dir+'_raw', class_list[i]), os.path.join(test_dir, new_class[i]))


train_dir = 'data/organmnist3d/train'
test_dir = 'data/organmnist3d/test'
datafolder_rename(train_dir, test_dir)