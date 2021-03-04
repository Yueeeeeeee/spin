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


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)


def train(model, train_loader, val_loader, export_root, num_epoch=100, resume=False):
    if resume:
        try:
            model.load_state_dict(torch.load(os.path.join(
                export_root, 'best_model.pth'), map_location='cpu'))
            print('Successfully loaded previous model, continue training...')
        except FileNotFoundError:
            print('Failed to load old model, continue training new model...')
    
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    criterio = nn.CrossEntropyLoss(ignore_index=-1)

    train_loss_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []
    val_iou_epoch = []

    best_loss, best_acc, best_iou = evaluate(model, val_loader)
    for epoch in range(num_epoch):
        model.train()
        train_loss_itr = []
        tqdm_dataloader = tqdm(train_loader)
        for i, (img, target, label) in enumerate(tqdm_dataloader):
            img = img.cuda()
            label_one_hot = target.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(img)
            loss = criterio(output, label.long())
            loss.backward()
            optimizer.step()
            train_loss_itr.append(loss.item())
            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch+1, np.mean(train_loss_itr)))
        
        lr_sheduler.step()
        train_loss_epoch.append(np.mean(train_loss_itr))

        val_loss, val_acc, val_iou = evaluate(model, val_loader)
        if val_acc + np.mean(val_iou) > best_acc + np.mean(best_iou):
            best_loss, best_acc, best_iou = val_loss, val_acc, val_iou
            if not os.path.exists(export_root):
                os.makedirs(export_root)
            print('Saving best model...')
            torch.save(model.state_dict(), export_root.joinpath('best_model.pth'))

        val_loss_epoch.append(val_loss)
        val_acc_epoch.append(val_acc)
        val_iou_epoch.append(val_iou)


def evaluate(model, data_loader):
    model.eval()
    val_loss_itr = []
    val_acc_itr = []
    val_iou_itr = []
    tqdm_dataloader = tqdm(data_loader)
    criterio = nn.CrossEntropyLoss(ignore_index=-1)
    for i, (img, target, label) in enumerate(tqdm_dataloader):
        with torch.no_grad():
            img = img.cuda()
            label_one_hot = target.cuda()
            label = label.cuda()

            output = model(img)
            pred = torch.argmax(output, dim=1)
            pred_onehot = to_one_hot(pred, 3).cuda()
            
            loss = criterio(output, label.long())
            acc = pixel_acc(pred, label)
            ious = compute_iou(pred_onehot, label_one_hot)
            
            val_loss_itr.append(loss.item())
            val_acc_itr.append(acc)
            val_iou_itr.append(ious)
            tqdm_dataloader.set_description('Evaluation, loss {:.3f}, acc {:.3f}, iou {:.3f} '.format(
                np.mean(val_loss_itr), np.mean(val_acc_itr), np.mean(val_iou_itr)))

    return np.mean(val_loss_itr), np.mean(val_acc_itr), np.mean(val_iou_itr, axis=0).tolist()


if __name__ == "__main__":
    fix_random_seed_as(12345)

    im_north_gt = Image.open('./data/CDL_2013_Champaign_north.tif')
    im_north_gt = np.array(im_north_gt).astype(int)
    im_north_gt[im_north_gt==1] = 0
    im_north_gt[im_north_gt==5] = 1
    im_north_gt[im_north_gt>1] = 2

    im_north = rasterio.open('./data/20130824_RE3_3A_Analytic_Champaign_north.tif').read()
    im_north = (im_north.reshape(-1, 5959*9425) - np.array(DATA_MEAN)) / np.array(DATA_STD)
    im_north = im_north.reshape(-1, 5959, 9425)

    train_dataset = TrainDataset(im_north[:, :, :8000], im_north_gt[:, :8000])
    val_dataset = TrainDataset(im_north[:, :, 8000:], im_north_gt[:, 8000:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    model_code = {'unet': UNet, 'resnet': ResNet_Deconv}
    model_type = input(
        'Input unet for U-Net, resnet for ResNet-FCN: ')
    model = model_code[model_type](3)
    export_root = Path('./model/' + model_type)

    train(model, train_loader, val_loader, export_root, num_epoch=200, resume=True)