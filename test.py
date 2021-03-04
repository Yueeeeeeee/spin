import rasterio
from rasterio.plot import show
from PIL import Image

import torch
import numpy as np
import torch.optim as optim
import random

import os
from tqdm import tqdm
from pathlib import Path
from models import *
from dataset import *
from utils import *


def test(model_type, model, img, export_root):
    try:
        model.load_state_dict(torch.load(os.path.join(
            export_root, 'best_model.pth'), map_location='cpu'))
        print('Successfully loaded previous model, testing...')
    except FileNotFoundError:
        print('Failed to load old model, terminating...')
        exit()

    model.cuda()
    model.eval()
    with torch.no_grad():
        img = img.cuda()
        output = model(img[:, :, :, :2000])
        pred1 = torch.argmax(output, dim=1)[:, :, :2000]
        
        output = model(img[:, :, :, 2000:4000])
        pred2 = torch.argmax(output, dim=1)[:, :, :2000]

        output = model(img[:, :, :, 4000:6000])
        pred3 = torch.argmax(output, dim=1)[:, :, :2000]

        output = model(img[:, :, :, 6000:8000])
        pred4 = torch.argmax(output, dim=1)[:, :, :2000]

        output = model(img[:, :, :, 8000:])
        pred5 = torch.argmax(output, dim=1)[:, :, :2000]

        # output = model(img[:, :, :, :1000])
        # pred1 = torch.argmax(output, dim=1)[:, :, :1000]
        
        # output = model(img[:, :, :, 1000:2000])
        # pred2 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 2000:3000])
        # pred3 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 3000:4000])
        # pred4 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 4000:5000])
        # pred5 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 5000:6000])
        # pred6 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 6000:7000])
        # pred7 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 7000:8000])
        # pred8 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 8000:9000])
        # pred9 = torch.argmax(output, dim=1)[:, :, :1000]

        # output = model(img[:, :, :, 9000:])
        # pred10 = torch.argmax(output, dim=1)

        pred = torch.cat((pred1, pred2, pred3, pred4, pred5), dim=-1)
        # pred = torch.cat((pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10), dim=-1)
        pred = pred[:, :5959, :9425].squeeze()
        print('Entire Output Shape:', pred.shape)

        img = np.zeros((pred.size(0), pred.size(1), 3), dtype=np.uint8)
        pred = pred.cpu().numpy()
        img[pred==0] = [255, 255, 0]
        img[pred==1] = [34, 139, 34]
        img[pred==2] = [128, 128, 128]

        image = Image.fromarray(img, 'RGB')
        image.save(model_type + '_test.png')


if __name__ == "__main__":
    im_south = rasterio.open('./data/20130824_RE3_3A_Analytic_Champaign_south.tif').read()
    blank_space = np.zeros((5, 6000, 9500))
    blank_space[:, :5959, :9425] = im_south
    im_south = blank_space
    im_south = (im_south.reshape(-1, 6000*9500) - np.array(DATA_MEAN)) / np.array(DATA_STD)
    im_south = torch.FloatTensor(im_south.reshape(-1, 6000, 9500)).unsqueeze(0)

    model_code = {'unet': UNet, 'resnet': ResNet_Deconv}
    model_type = input(
        'Input unet for U-Net, resnet for ResNet-FCN: ')
    model = model_code[model_type](3)
    export_root = Path('./model/' + model_type)

    test(model_type, model, im_south, export_root)