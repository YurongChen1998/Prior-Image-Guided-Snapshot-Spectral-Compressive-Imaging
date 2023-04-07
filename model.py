##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import torch
from func import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def admm_denoise(meas, Phi, data_truth, ref_img, args):
    #-------------- Initialization --------------#
    if args.x0 is None:
        x0 = At(meas, Phi)
    iter_max = [args.iter_max] * len(args.sigma)
    ssim_all = []
    psnr_all = []
    k = 0
    show_iqa = True
    noise_estimate = True
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0.to(device)
    theta = x0.to(device)
    b = torch.zeros_like(x0).to(device)
    gamma = 0.03

    upper_bound_list = []
    true_value_list = []

    # ---------------- Iteration ----------------#
    for idx, noise_level in enumerate(args.sigma):
        for iter in range(iter_max[idx]):
            meas_b = A(theta.to(device)+b.to(device), Phi)
            x = (theta + b) + args.lambda_*(At((meas - meas_b)/(Phi_sum + gamma), Phi))
            x1 = shift_back(x-b, step=2)

            if args.denoiser == 'TV':
                theta = TV_minimization(x1, ref_img, args.tv_weight, args.tv_iter_max)
                
            # --------------- Evaluation ---------------#
            if show_iqa and data_truth is not None:
                ssim_all = calculate_ssim(data_truth*0.9, theta)
                psnr_all = calculate_psnr(data_truth*0.9, theta)
                if (k + 1) % 1 == 0:
                    print('  ADMM-{0} iteration {1: 3d}, '
                          'PSNR {2:2.2f} dB.'.format(args.denoiser.upper(), k + 1, psnr_all),
                          'SSIM:{:2.3f}.'.format(ssim_all))

            theta = shift(theta, step=2)
            b = b - (x.to(device) - theta.to(device))
            k += 1
    return theta, psnr_all


def gap_denoise(meas, Phi, data_truth, ref_img, args):
    pass