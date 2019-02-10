# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:05:15 2019

@author: Andrew Teta
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
from collections import Counter
import lab2_funcs as lf

# ========== Grayscale Conversion =========== #
#im1 = Image.open('images/test01.jpg')
## convert image to grayscale
#gray1 = lf.grayscale(im1)
## Output grayscale image to file
#Image.fromarray(gray1.astype(np.uint8)).save('figs/test01_gray.jpg')
#im2 = Image.open('images/test02.jpg')
## convert image to grayscale
#gray2 = lf.grayscale(im2)
## Output grayscale image to file
#Image.fromarray(gray2.astype(np.uint8)).save('figs/test02_gray.jpg')

# ========== Histogram Equalization =========== #
# lc1
im = Image.open('images/lc1.jpg')
contrast_enhanced_lc1, orig_hist_lc1, post_hist_lc1, remap1, im_out1 = lf.histEQ(im)
Image.fromarray(im_out1.astype(np.uint8)).save('figs/lc1_gray.jpg')
Image.fromarray(contrast_enhanced_lc1.astype(np.uint8)).save('figs/lc1_ce.jpg')
plt.figure(dpi=170)
plt.plot(orig_hist_lc1)
plt.title('Intensity Histogram')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.savefig('figs/lc1_hist_bef')
plt.figure(dpi=170)
plt.plot(post_hist_lc1)
plt.title('Intensity Histogram')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.savefig('figs/lc1_hist_aft')
plt.figure(dpi=170)
plt.plot(remap1)
plt.title('Remapping Function')
plt.ylabel('Input')
plt.xlabel('Output')
plt.savefig('figs/lc1_remap')
# lc2
im = Image.open('images/lc2.jpg')
contrast_enhanced_lc2, orig_hist_lc2, post_hist_lc2, remap2, im_out2 = lf.histEQ(im)
Image.fromarray(im_out2.astype(np.uint8)).save('figs/lc2_gray.jpg')
Image.fromarray(contrast_enhanced_lc2.astype(np.uint8)).save('figs/lc2_ce.jpg')
plt.figure(dpi=170)
plt.plot(orig_hist_lc2)
plt.title('Intensity Histogram')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.savefig('figs/lc2_hist_bef')
plt.figure(dpi=170)
plt.plot(post_hist_lc2)
plt.title('Intensity Histogram')
plt.ylabel('Number of pixels')
plt.xlabel('Intensity')
plt.savefig('figs/lc2_hist_aft')
plt.figure(dpi=170)
plt.plot(remap2)
plt.title('Remapping Function')
plt.ylabel('Input')
plt.xlabel('Output')
plt.savefig('figs/lc2_remap')

# ========== Sobel Edge Detection =========== #
#im = Image.open('images/test01.jpg')
#edges = lf.Sobel(im, 80)
#Image.fromarray(edges).show()

# ========== Scale down by 2, then up by 2 and take the difference =========== #
#im = Image.open('images/test01.jpg')
#smallImage = lf.scaleDown(im, 2)
#Image.fromarray(smallImage.astype(np.uint8)).save('figs/test01_down2.jpg')
#downUp = lf.upScale(Image.open('figs/test01_down2.jpg'), 2)
#Image.fromarray(downUp.astype(np.uint8)).save('figs/test01_down2_up2.jpg')
#diff1 = lf.difference(lf.grayscale(im), downUp)
#Image.fromarray(diff1).show()
#Image.fromarray(diff1.astype(np.uint8)).save('figs/test01_downUp2_diff.jpg')

# ========== Scale down by 4, then up by 4 and take the difference =========== #
#im = Image.open('images/test01.jpg')
#smallImage1 = lf.scaleDown(im, 4)
#Image.fromarray(smallImage1.astype(np.uint8)).save('figs/test01_down4.jpg')
#downUp1 = lf.upScale(Image.open('figs/test01_down4.jpg'), 4)
#Image.fromarray(downUp1.astype(np.uint8)).save('figs/test01_down4_up4.jpg')
#diff2 = lf.difference(lf.grayscale(im), downUp1)
#Image.fromarray(diff2).show()
#Image.fromarray(diff2.astype(np.uint8)).save('figs/test01_downUp4_diff.jpg')

print ('done')