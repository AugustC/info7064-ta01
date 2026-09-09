import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_orientation_filters(n=6, ksize=31,  sigma=1, size=0):
    kernels = []
    orientations = np.arange(0, np.pi, np.pi / n)
    for orientation in orientations:
        kernel = cv2.getGaborKernel((ksize,ksize), sigma, orientation, ksize, size)
        kernels.append(kernel)
    return kernels

def get_circular_filters(ksize = 32, radius=15):
    sigma = radius/4
    x, y = np.meshgrid(np.arange(-ksize + 1, ksize + 1),np.arange(-ksize + 1, ksize + 1))
    normalizer = 1 / (np.pi * (sigma ** 4))
    exponent = -(x**2 + y**2) / (2 * (sigma ** 2))
    kernel = -normalizer * (1 + exponent) * np.exp(exponent)
    kernel -= kernel.sum() / (ksize * ksize)
    return kernel, -kernel

def apply_kernels(src, kernels):
    filtered_imgs = [src]
    for k in kernels:
        f_img = cv2.filter2D(src = src, ddepth = -1, kernel = k)
        f_img = cv2.normalize(f_img, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        filtered_imgs.append(f_img)
    return filtered_imgs
            