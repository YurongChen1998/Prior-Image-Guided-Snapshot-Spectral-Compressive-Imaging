##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import time
import torch
import argparse
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import gap_denoise, admm_denoise
from torchmetrics import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)


#-----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ADMM', help="Select GAP or ADMM")
parser.add_argument('--lambda_', default=1, help="_lambda is the regularization factor")
parser.add_argument('--denoiser', default='TV', help="Select which denoiser: Total Variation (TV)")
parser.add_argument('--accelerate', default=True, help="Acclearted version of GAP or not")
parser.add_argument('--iter_max', default=30, help="Maximum number of iterations")
parser.add_argument('--tv_weight', default=25, help="TV denoising weight (larger for smoother but slower)")
parser.add_argument('--tv_iter_max', default=25, help="TV denoising maximum number of iterations each")
parser.add_argument('--x0', default=None, help="The initialization data point")
parser.add_argument('--sigma', default=[130, 130, 130, 130, 130, 130], help="The noise levels")
args = parser.parse_args()
#------------------------------------------------------------------#


#----------------------- Data Configuration -----------------------#
dataset_dir = './Dataset/CAVE_Dataset/Orig_data/'
results_dir = 'results'
data_name = 'scene01'
matfile = dataset_dir + data_name + '.mat'
h, w, nC, step, beta = 512, 512, 28, 2, 0.9
data_truth = torch.from_numpy(sio.loadmat(matfile)['data_slice']) / 65536

data_truth_shift = torch.zeros((h, w + step*(nC - 1), nC))
for i in range(nC):
    data_truth_shift[:, i*step:i*step+h, i] = data_truth[:, :, i]
data_truth_shift = data_truth_shift * beta

ref_matfile = './Dataset/CAVE_Dataset/Bayer_RGB/' + data_name + '_rgb.mat'
ref_img = torch.from_numpy(sio.loadmat(ref_matfile)['rgb_recon']).to(device)
ref_img_1 = torch.unsqueeze(ref_img[:, :, 2], 2)#.repeat(1, 1, 10)
ref_img_2 = torch.unsqueeze(ref_img[:, :, 1], 2)#.repeat(1, 1, 8)
ref_img_3 = torch.unsqueeze(ref_img[:, :, 0], 2)#.repeat(1, 1, 10)
ref_img = torch.cat((ref_img_1, ref_img_2, ref_img_3), dim=2)
ref_img = F.interpolate(ref_img, size = [nC], mode='linear')
'''
N = ref_img.shape
idx = torch.arange(1, N[0]+1)
idx[-1] = N[0]-1
ir = torch.arange(1, N[1]+1)
ir[-1] = N[1]-1

x1 = data_truth[:,ir,:] - data_truth
x2 = data_truth[idx,:,:] - data_truth
tv = torch.abs(x1) + torch.abs(x2)
sio.savemat('./result/tv.mat', {'tv':tv.cpu().numpy()})
'''
ref_img = ref_img * (1 - beta) 
#------------------------------------------------------------------#


#----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + step*(nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
mask_256 = torch.from_numpy(sio.loadmat('./Dataset/CAVE_Dataset/mask512.mat')['mask'])
for i in range(nC):
    mask_3d[:, i*step:i*step+h, i] = mask_256
Phi = mask_3d
meas = torch.sum(Phi * data_truth_shift, 2)

plt.figure()
plt.imshow(meas,cmap='gray')
plt.savefig('result/meas.png')
#------------------------------------------------------------------#


begin_time = time.time()
if args.method == 'GAP':
    pass
        
elif args.method == 'ADMM':
    recon, psnr_all = admm_denoise(meas.to(device), Phi.to(device), data_truth.to(device), ref_img.to(device), args)
    end_time = time.time()
    recon = shift_back(recon, step=2)
    
    sam = SpectralAngleMapper()
    vrecon = recon.double().cpu()
    sam = sam(torch.unsqueeze(vrecon.permute(2, 0, 1), 0).double(), torch.unsqueeze(data_truth.permute(2, 0, 1), 0).double())
    print('ADMM, SAM {:2.3f}, running time {:.1f} seconds.'.format(sam, end_time - begin_time))


sio.savemat('./result/result.mat', {'img':recon.cpu().numpy()})
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(recon[:,:,(i+1)*3], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('./result/result.png')
