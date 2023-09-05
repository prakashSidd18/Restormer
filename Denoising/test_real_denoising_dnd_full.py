## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/DND/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Real_Denoising/DND/full/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/real_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

israw = False
eval_version="1.0"

# Load info
infos = h5py.File(os.path.join(args.input_dir, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']
factor = 512
# Process data
with torch.no_grad():
    for i in tqdm(range(50)):
        filename = '%04d.mat'%(i+1)
        filepath = os.path.join(args.input_dir, 'images_srgb', filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        # ref = bb[0][i]
        # boxes = np.array(info[ref]).T

        h, w = Inoisy.shape[0], Inoisy.shape[1]

        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        Inoisy_new = np.zeros((H, W, Inoisy.shape[2]))
        Inoisy_new[:h, :w, :] = Inoisy

        nboxes, H_new, W_new = 0, 0, 0
        boxes = []
        while H_new + factor <= H:
            W_new = 0
            while W_new + factor <= W:
                boxes.append([H_new, W_new, H_new+factor, W_new+factor])
                W_new += factor
            H_new += factor

        nboxes = len(boxes)
        boxes = np.array(boxes)

        Idenoised = np.zeros((nboxes,), dtype=np.object)
        Idenoised_full = np.zeros(Inoisy_new.shape)
        for k in range(nboxes):
            idx = [int(boxes[k,0]),int(boxes[k,2]),int(boxes[k,1]),int(boxes[k,3])]
            noisy_patch = torch.from_numpy(Inoisy_new[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda().float()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            Idenoised[k] = restored_patch
            Idenoised_full[idx[0]:idx[1],idx[2]:idx[3],:] = restored_patch

            # if args.save_images:
            #     save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
            #     denoised_img = img_as_ubyte(restored_patch)
            #     utils.save_img(save_file, denoised_img)
            #
            #     save_file = os.path.join(result_dir_png, '%04d_%02d_noisy.png' % (i + 1, k + 1))
            #     noisy_img = img_as_ubyte(noisy_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy())
            #     utils.save_img(save_file, noisy_img)

        Idenoised_full = Idenoised_full[:h, :w, :]
        if args.save_images:
            save_file = os.path.join(result_dir_png, '%04d.png' % (i + 1))
            denoised_img_full = img_as_ubyte(Idenoised_full)
            utils.save_img(save_file, denoised_img_full)

        # save denoised data
        # sio.savemat(os.path.join(result_dir_mat, filename),
        #             {"Idenoised": Idenoised,
        #              "israw": israw,
        #              "eval_version": eval_version},
        #             )
