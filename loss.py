from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).cuda())
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()
        
        self.xconv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1).cuda()
        self.xconv.bias.data.zero_()
        self.xconv.weight.data[0,0,:,:] = torch.FloatTensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).cuda()
        for param in self.xconv.parameters():
            param.requires_grad = False
            
        self.yconv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1).cuda()
        self.yconv.bias.data.zero_()
        self.yconv.weight.data[0,0,:,:] = torch.FloatTensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]).cuda()
        for param in self.yconv.parameters():
            param.requires_grad = False

    def MMDcompute(self, x, y, alpha=1):
            
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()),  torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        rx1 = (xx.diag().unsqueeze(0).expand_as(torch.Tensor(y.size(0), x.size(0))))
        ry1 = (yy.diag().unsqueeze(0).expand_as(torch.Tensor(x.size(0), y.size(0))))

        K = (torch.exp(- 0.5*alpha * (rx.t() + rx - 2*xx)) + torch.exp(- 0.1*alpha * (rx.t() + rx - 2*xx)) +
            torch.exp(- 0.05*alpha * (rx.t() + rx - 2*xx))) / 3
        L = (torch.exp(- 0.5*alpha * (ry.t() + ry - 2*yy)) + torch.exp(- 0.1*alpha * (ry.t() + ry - 2*yy)) +
            torch.exp(- 0.05*alpha * (ry.t() + ry - 2*yy))) / 3
        P = (torch.exp(- 0.5*alpha * (rx1.t() + ry1 - 2*zz)) + torch.exp(- 0.1*alpha * (rx1.t() + ry1 - 2*zz)) +
            torch.exp(- 0.05*alpha * (rx1.t() + ry1 - 2*zz))) / 3

        beta1 = (1. / (x.size(0) * x.size(0)))
        beta2 = (1. / (y.size(0) * y.size(0)))
        gamma = (2. / (x.size(0) * y.size(0)))

        return beta1 * torch.sum(K) + beta2 * torch.sum(L) - gamma * torch.sum(P)

    def rgb2gray(self, img):
        (batch, channel, height, width) = img.size()
        gImg = Variable(torch.ones(batch, 1, height, width)).cuda()
        
        for i in range(batch):
            grayimg = 0.2989 * img[i,0,:,:] + 0.5870 * img[i,1,:,:] + 0.1140 * img[i,2,:,:]
            gImg[i,0,:,:] = grayimg
        
        return gImg

    def forward(self, im1, im2):
        im1g = self.rgb2gray(im1)
        im2g = self.rgb2gray(im2)
        
        im1gx = self.xconv(im1g)
        im1gy = self.yconv(im1g)
        
        im2gx = self.xconv(im2g)
        im2gy = self.yconv(im2g)
        
        (batch, channel, height, width) = im1.size()
        
        im1xd = F.softmax(im1gx.view(-1, height*width), dim=1)
        im2xd = F.softmax(im2gx.view(-1, height*width), dim=1)
        
        im1yd = F.softmax(im1gy.view(-1, height*width), dim=1)
        im2yd = F.softmax(im2gy.view(-1, height*width), dim=1)
        
        loss = self.MMDcompute(im1xd, im2xd) + self.MMDcompute(im1yd, im2yd)
        
        return loss


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.mmd_loss = MMDLoss()

    def forward(self, preds, labels):
        # Order: r-g-b
        pred_r, pred_g, pred_b = preds['r'], preds['g'], preds['b']
        label_r, label_g, label_b = labels['r'], labels['g'], labels['b']
        
        loss_r = self.ssim_loss(pred_r, label_r) + self.mmd_loss(pred_r, label_r) * 0.5
        loss_g = self.ssim_loss(pred_g, label_g)
        loss_b = self.ssim_loss(pred_b, label_b) * 0.8 + self.L1_loss(pred_b, label_b) + self.mmd_loss(pred_b, label_b)

        loss_total = loss_r + loss_g + loss_b

        return loss_total

