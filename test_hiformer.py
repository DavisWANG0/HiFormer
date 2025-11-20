"""
Testing script for HiFormer
"""

import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.hiformer import HiFormer

import lightning.pytorch as pl


class HiFormerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = HiFormer(
            inp_channels=3,
            out_channels=3,
            dim=24,
            num_blocks=[4, 4, 8],
            heads=[1, 2, 4],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            window_size=4
        )
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)


def test_Denoise(net, dataset, sigma=15, testopt=None):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain", testopt=None):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("Task: %s, PSNR: %.2f, SSIM: %.4f" % (task, psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", 
                        help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", 
                        help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", 
                        help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", 
                        help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="hiformer-epoch-499.ckpt", 
                        help='checkpoint filename')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/hiformer/" + testopt.ckpt_name

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("=" * 50)
    print(f"Loading checkpoint: {ckpt_path}")
    print("=" * 50)

    # Load model
    net = HiFormerModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        print("\n" + "=" * 50)
        print("Testing Denoising")
        print("=" * 50)
        for testset, name in zip(denoise_tests, denoise_splits):
            print(f'\nTesting {name} with Sigma=15...')
            test_Denoise(net, testset, sigma=15, testopt=testopt)

            print(f'Testing {name} with Sigma=25...')
            test_Denoise(net, testset, sigma=25, testopt=testopt)

            print(f'Testing {name} with Sigma=50...')
            test_Denoise(net, testset, sigma=50, testopt=testopt)
            
    elif testopt.mode == 1:
        print("\n" + "=" * 50)
        print("Testing Rain Streak Removal")
        print("=" * 50)
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print(f'\nTesting {name} rain streak removal...')
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain", testopt=testopt)
            
    elif testopt.mode == 2:
        print("\n" + "=" * 50)
        print("Testing Dehazing (SOTS)")
        print("=" * 50)
        dehaze_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, dehaze_set, task="dehaze", testopt=testopt)
        
    elif testopt.mode == 3:
        print("\n" + "=" * 50)
        print("Testing All Tasks")
        print("=" * 50)
        
        # Denoise
        print("\n--- Denoising ---")
        for testset, name in zip(denoise_tests, denoise_splits):
            print(f'\nTesting {name} with Sigma=15...')
            test_Denoise(net, testset, sigma=15, testopt=testopt)
            print(f'Testing {name} with Sigma=25...')
            test_Denoise(net, testset, sigma=25, testopt=testopt)
            print(f'Testing {name} with Sigma=50...')
            test_Denoise(net, testset, sigma=50, testopt=testopt)

        # Derain
        print("\n--- Rain Streak Removal ---")
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print(f'\nTesting {name}...')
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain", testopt=testopt)

        # Dehaze
        print("\n--- Dehazing ---")
        test_Derain_Dehaze(net, derain_set, task="dehaze", testopt=testopt)

    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)

