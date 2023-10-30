import math
import torch
import pprint
import os
import json

XWIDTH = 72
DIGITS = 3
DIGITS10 = math.pow(10, DIGITS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pp = pprint.PrettyPrinter(indent=4)


ROOT_PATH = 'DRM/SAVE/EXPS/'


def shift_euler(eulers, shift):
    angular_shift =  (2 * torch.pi / XWIDTH) * shift
    eulers = eulers.clone()
    eulers[:, 0] = (eulers[:, 0] + angular_shift)  % (2 * torch.pi)
    return eulers  

def mkdir(EXP_PATH):
    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH, exist_ok=True)


def save_data_to_file(data, EXP_PATH, file_path, indent=4):
    with open(os.path.join(EXP_PATH, file_path), 'w') as f:
        json.dump(data, f, indent=indent)


def normalize_images(images, amean=128.0, astd=64.0):
    mean = torch.mean(images, dim=(1, 2, 3), keepdim=True)
    std = torch.std(images, dim=(1, 2, 3), keepdim=True)    
    scale_mean = amean 
    scale_std = astd / std
    normalized_images = (images - mean) * scale_std + scale_mean
    return normalized_images.clamp(0, 255)        


def augment_shift(X, Y, idx_shift):
    _, _,s0,s1 = X.shape
    Xret = torch.roll(X, shifts=idx_shift, dims=-1)
    angular_shift = (idx_shift / s1) * torch.pi * 2
    Yret = Y.clone()
    Yret[:, 0] = (Y[:, 0] + angular_shift) % (2 * torch.pi)
    return Xret, Yret



def get_random_samples(X, Y, N, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    assert X.size(0) == Y.size(0), "First dimension of X and Y should be the same"
    indices = torch.randperm(X.size(0))
    sample_indices = indices[:N]
    X_samples = X[sample_indices]
    Y_samples = Y[sample_indices]
    return X_samples, Y_samples


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def disp(your_image_data, width=6, height=2, cmap='magma', vmin=0, vmax=255, save_path=None):
    # Set the desired width and height of the plot in inches
    # Create the figure and set its size
    plt.figure(figsize=(width, height))

    # Display the scaled image as a heatmap using matplotlib
    plt.imshow(your_image_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')  # Hide the axis ticks and labels

    # Save the plot to an image file if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()
