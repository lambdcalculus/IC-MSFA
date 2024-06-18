import torch
import os
from torch.utils.data import Dataset
from scipy.io import loadmat

from typing import Optional
from collections.abc import Callable

class MatDataset(Dataset):
    """Reads MATLAB files into a dataset.

    Args:
        img_dir (str): directory of the .mat files
        transform_x (optional Callable): how to create the input images from the base files
        transform_y (optional Callable): how to create the output images from the base files
    """

    def __init__(self, img_dir: str, transform_x: Optional[Callable] = None, transform_y: Optional[Callable] = None):
        self.img_dir = img_dir
        self.files = [name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name))]     
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = os.path.join(self.img_dir, self.files[i])
        data = loadmat(file)
        image = torch.Tensor(data['img']).transpose(0, 2) # C x W x H

        x, y = image, image
        with torch.no_grad():
            if self.transform_x:
                x = self.transform_x(x)
            if self.transform_y:
                y = self.transform_y(y)
        return x, y
