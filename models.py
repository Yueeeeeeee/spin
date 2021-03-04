import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d


pretrained_models = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
}
encoder_out_chnnel={
    'resnet18': 512,
    'resnet34': 1024,
    'resnet50': 2048,
}


class UNet(nn.Module):
    def __init__(self, num_classes, scaling_factor=4):
        super().__init__()
        k = scaling_factor
        self.layer1 = nn.Sequential(nn.Conv2d(5, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, 64 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer2 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(64 // k, 128 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128 // k, 128 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer3 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(128 // k, 256 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(256 // k, 256 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer4 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(256 // k, 512 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(512 // k, 512 // k, 3, padding=1), nn.ReLU()
                                    )
        self.layer5 = nn.Sequential(nn.MaxPool2d(2),
                                    nn.Conv2d(512 // k, 1024 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(1024 // k, 1024 // k, 3, padding=1), nn.ReLU()
                                    # nn.Conv2d(1024, 512, 2),
                                    )
        self.deconv1 = nn.ConvTranspose2d(1024 // k, 512 // k, 2, stride=2)
        self.layer6 = nn.Sequential(nn.Conv2d(1024 // k, 512 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(512 // k, 512 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv2 = nn.ConvTranspose2d(512 // k, 256 // k, 2, stride=2)
        self.layer7 = nn.Sequential(nn.Conv2d(512 // k, 256 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(256 // k, 256 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv3 = nn.ConvTranspose2d(256 // k, 128 // k, 2, stride=2)
        self.layer8 = nn.Sequential(nn.Conv2d(256 // k, 128 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(128 // k, 128 // k, 3, padding=1), nn.ReLU()
                                    )
        self.deconv4 = nn.ConvTranspose2d(128 // k, 64 // k, 2, stride=2)
        self.layer9 = nn.Sequential(nn.Conv2d(128 // k, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, 64 // k, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(64 // k, num_classes, 1)
                                    )

    def forward(self, x):
        en_x1 = self.layer1(x)
        en_x2 = self.layer2(en_x1)
        en_x3 = self.layer3(en_x2)
        en_x4 = self.layer4(en_x3)
        en_x5 = self.layer5(en_x4)
        de_h1 = self.deconv1(en_x5)

        h1, h2, w1, w2 = self.resize_shape(en_x4.shape, de_h1.shape)
        h2 = self.layer6(torch.cat([en_x4[:, :, h1:h2, w1:w2], de_h1], dim=1))
        de_h2 = self.deconv2(h2)

        h1, h2, w1, w2 = self.resize_shape(en_x3.shape, de_h2.shape)
        h3 = self.layer7(torch.cat([en_x3[:, :, h1:h2, w1:w2], de_h2], dim=1))
        de_h3 = self.deconv3(h3)

        h1, h2, w1, w2 = self.resize_shape(en_x2.shape, de_h3.shape)
        h4 = self.layer8(torch.cat([en_x2[:, :, h1:h2, w1:w2], de_h3], dim=1))
        de_h4 = self.deconv4(h4)

        h1, h2, w1, w2 = self.resize_shape(en_x1.shape, de_h4.shape)
        h5 = self.layer9(torch.cat([en_x1[:, :, h1:h2, w1:w2], de_h4], dim=1))
        return h5

    def resize_shape(self, shape1, shape2):
        hh1, ww1 = shape1[-2], shape1[-1]
        hh2, ww2 = shape2[-2], shape2[-1]
        h1 = int(hh1 / 2 - hh2 / 2)
        h2 = hh2 + h1
        w1 = int(ww1 / 2 - ww2 / 2)
        w2 = ww2 + w1
        return h1, h2, w1, w2


class ResNet_Deconv(nn.Module):
    def __init__(self, num_classes, retrain=True, backbone='resnet50'):
        super().__init__()
        self.n_class = num_classes
        self.encoder = self.load_encoder(backbone)
        self.encoder[0] = Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if retrain:
            for params in self.encoder.parameters():
                params.requires_grad = True
        else:
            for params in self.encoder.parameters():
                params.requires_grad = False

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_out_chnnel[backbone], 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        out_decoder = self.deconv5(x)
        score = self.classifier(out_decoder)
        return score

    def load_encoder(self, backbone):
        pretrained_net = pretrained_models[backbone](pretrained=True)
        encoder = nn.Sequential()

        for idx, layer in enumerate(pretrained_net.children()):
            if isinstance(layer, nn.Linear) == False and isinstance(layer, nn.AdaptiveAvgPool2d) == False:
                encoder.add_module(str(idx), layer)

        return encoder


if __name__ == "__main__":
    resnet = ResNet_Deconv(3)
    unet = UNet(3)
    x = torch.randn(1,5,512,512)
    print(resnet(x).shape)
    print(unet(x).shape)