import rasterio
from rasterio.plot import show
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils import *


class TrainDataset(Dataset):
    def __init__(self, img, gt, n_class=3, window_size=[500, 500], stride=[250, 250], transformation=None):
        self.img = img
        self.gt = gt
        self.n_class = n_class
        self.window_size = window_size
        self.stride = stride
        self.transform = transformation
        
        self.length_x = int(np.ceil((img.shape[1] - self.window_size[0]) / self.stride[0]) + 1)
        self.length_y = int(np.ceil((img.shape[2] - self.window_size[1]) / self.stride[1]) + 1)
        self.length = self.length_x * self.length_y

        if transformation is None:
            self.transform = [
                RandomFlip(0.5),
                RandomCrop([320, 320]),
                ]
        else:
            self.transform = transformation

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x_position = idx // self.length_x
        y_position = idx % self.length_y

        x_start_index = x_position * self.stride[0]
        x_end_index = np.min((self.img.shape[1], x_position*self.stride[0]+self.window_size[0]))
        y_start_index = y_position * self.stride[1]
        y_end_index = np.min((self.img.shape[2], y_position*self.stride[1]+self.window_size[1]))

        img = self.img[:, x_start_index:x_end_index, y_start_index:y_end_index].copy()
        label = self.gt[x_start_index:x_end_index, y_start_index:y_end_index].copy()
        img, label = self.padding_img(img, label)
        if self.transform is not None:
            for trans in self.transform:
                img, label = trans(img, label)

        # create one-hot encoding
        h, w = label.shape[0], label.shape[1]
        target = torch.zeros(self.n_class, h, w).float()
        for c in range(self.n_class):
            target[c][c==label] = 1

        return torch.Tensor(img.copy()), torch.Tensor(target), torch.Tensor(label.astype(int))

    def padding_img(self, img, label):
        assert img.shape[1:] == label.shape
        if label.shape[0] == self.window_size[0] and label.shape[1] == self.window_size[1]:
            return img, label

        img_array = np.zeros((5, self.window_size[0], self.window_size[1]))
        out_array = np.ones((self.window_size[0], self.window_size[1])) * 2
        img_array[:, :img.shape[1], :img.shape[2]] = img
        out_array[:label.shape[0], :label.shape[1]] = label
        return img_array, out_array