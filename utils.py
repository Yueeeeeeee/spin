import numpy as np
import torch
import torch.nn.functional as F


DATA_MEAN = [[5188.78934685], [4132.74818647], [2498.34747813], [3689.04702811], [11074.86217434]]
DATA_STD = [[1482.89729162], [1447.21062441], [1384.91231294], [1149.82168184], [2879.24827197]]


def pixel_acc(pred, label):
    correct = torch.sum(pred==label).item()
    N = label[label>=0].view(-1).shape[0]
    return correct / N


def compute_iou(pred, target):
    ious = torch.zeros(pred.shape[1]).cuda()
    intersection = torch.sum(pred * target, dim=[0,2,3]) # intersection every class
    union = torch.sum(pred, dim=[0,2,3]) + torch.sum(target, dim=[0,2,3]) - intersection
    ious[union!=0] = (intersection[union!=0] / union[union!=0])
    ious[union==0] = float('nan')
    return ious.tolist()


def to_one_hot(label, num_class):
    label_one_hot = torch.eye(num_class)[label]
    return label_one_hot.permute([0,3,1,2])


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    try:
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))
    except:
        c, h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(5, h//nrows, nrows, -1, ncols)
                   .swapaxes(2,3)
                   .reshape(-1, 5, nrows, ncols))
    

class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        p = np.random.randn()
        if p < (self.prob / 2):
            img = img[:, ::-1, :]
            label = label[::-1, :]
        elif p < self.prob:
            img = img[:, :, ::-1]
            label = label[:, ::-1]
        return img, label


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, img, label):
        start_x = int(np.random.choice(img.shape[1] - self.output_size[0], 1))
        start_y = int(np.random.choice(img.shape[2] - self.output_size[1], 1))

        img = img[:, start_x:start_x+self.output_size[0], start_y:start_y+self.output_size[1]]
        label = label[start_x:start_x+self.output_size[0], start_y:start_y+self.output_size[1]]       
        return img, label