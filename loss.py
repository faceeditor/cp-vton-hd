from torchvision import models
import torch
import torch.nn as nn


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT2, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, opt):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.opt = opt

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gxup = Gx[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gxdown = Gx[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gxleft = Gx[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gxright = Gx[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        Gycenter = Gy[:, 1:self.opt.fine_height - 1, 1:self.opt.fine_width - 1]
        Gyup = Gy[:, 0:self.opt.fine_height - 2, 1:self.opt.fine_width - 1]
        Gydown = Gy[:, 2:self.opt.fine_height, 1:self.opt.fine_width - 1]
        Gyleft = Gy[:, 1:self.opt.fine_height - 1, 0:self.opt.fine_width - 2]
        Gyright = Gy[:, 1:self.opt.fine_height - 1, 2:self.opt.fine_width]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))
