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
#im = Image.open('images/test01.jpg')
## convert image to grayscale
#gray1 = lf.grayscale(im)
#im = Image.open('images/test02.jpg')
## convert image to grayscale
#gray1 = lf.grayscale(im)
## Output grayscale image to file
##Image.fromarray(gray1).save('figs/test01_out.jpg')
#im = Image.open('images/lc1.jpg')
#contrast_enhanced_lc1, orig_hist_lc1, post_hist_lc1 = lf.histEQ(im)
#im = Image.open('images/lc2.jpg')
#contrast_enhanced_lc2, orig_hist_lc2, post_hist_lc2 = lf.histEQ(im)

# ========== Sobel Edge Detection =========== #
#im = Image.open('images/test01.jpg')
#edges = lf.Sobel(im, 80)
#Image.fromarray(edges).show()

# ========== Scale down by 2, then up by 2 and take the difference =========== #
im = Image.open('images/test01.jpg')
smallImage = lf.scaleDown(im, 2)
Image.fromarray(smallImage.astype(np.uint8)).save('figs/test01_down2.jpg')
downUp = lf.upScale(Image.open('figs/test01_down2.jpg'), 2)
Image.fromarray(downUp.astype(np.uint8)).save('figs/test01_down2_up2.jpg')
diff1 = lf.difference(smallImage, downUp)
Image.fromarray(diff1).show()
Image.fromarray(diff1.astype(np.uint8)).save('figs/test01_downUp2_diff.jpg')

print ('done')