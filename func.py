##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import cv2
import math
import torch
import numpy as np
from pytorch_msssim import ssim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def A(data, Phi):
    return torch.sum(data * Phi, 2)


def At(meas, Phi):
    meas = torch.unsqueeze(meas, 2).repeat(1, 1, Phi.shape[2])
    return meas * Phi


def shift(inputs, step):
    [h, w, nC] = inputs.shape
    output = torch.zeros((h, w+(nC - 1)*step, nC)).to(device)
    for i in range(nC):
        output[:, i*step : i*step + w, i] = inputs[:, :, i]
    del inputs
    return output


def shift_back(inputs, step):
    [h, w, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = torch.roll(inputs[:, :, i], (-1)*step*i, dims=1)
    output = inputs[:, 0 : w - step*(nC - 1), :]
    return output


def ssim_(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    if not data.dtype == recon.dtype:
        data, recon = data.float(), recon.float()
        
    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim_(data, recon)
    elif data.ndim == 3:
        return ssim(torch.unsqueeze(data, 0).permute(3, 0, 1, 2), torch.unsqueeze(recon, 0).permute(3, 0, 1, 2), data_range=1).data
     

def calculate_psnr(data, recon):
    mse = torch.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.
    return 20 * torch.log10(Pixel_max / torch.sqrt(mse))
    
    
def shrink(x, _lambda):
    u = torch.sign(x) * torch.clamp(x - 1/_lambda, 0)
    return u


def calculate_tv(x):
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1

    x1 = x[:,ir,:] - x
    x2 = x[idx,:,:] - x
    tv = torch.abs(x1) + torch.abs(x2)
    return torch.mean(torch.sum(tv, 2))

def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = torch.arange(-1, N[0]-1)
    iux[0] = 0
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = torch.arange(-1, N[1]-1)
    il[0] = 0
    p1 = torch.zeros_like(x)
    p2 = torch.zeros_like(x)
    divp = torch.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*torch.sqrt(torch.sum(z1**2 + z2**2, 2))
        denom_3d = torch.unsqueeze(denom_2d, 2).repeat(1, 1, N[2])
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda
    return u

def TV_minimization(x, y, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = torch.arange(-1, N[0]-1)
    iux[0] = 0
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = torch.arange(-1, N[1]-1)
    il[0] = 0
    p1 = torch.zeros_like(x)
    p2 = torch.zeros_like(x)
    p3 = torch.zeros_like(x)
    p4 = torch.zeros_like(x)
    divp = torch.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - (x - 0.95*10*y) * _lambda
        z1 = 0.5*(z[:,ir,:] - z)
        z2 = 0.5*(z[idx,:,:] - z)
        z3 = 0.5*(z - z[:,il,:])
        z4 = 0.5*(z - z[iux,:,:])
        denom_2d = 1 + dt*torch.sqrt(torch.sum(z1**2 + z2**2 + z3**2 + z4**2, 2))
        denom_3d = torch.unsqueeze(denom_2d, 2).repeat(1, 1, N[2])
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        p3 = (p3+dt*z3)/denom_3d
        p4 = (p4+dt*z4)/denom_3d
        divp = p1-p1[:,il,:] + p2-p2[iux,:,:] + p3[:,ir,:]-p3 + p4[idx,:,:]-p4
    u = x - divp/_lambda
    return u
