# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:05:15 2019

@author: Andrew Teta
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from collections import Counter

def grayscale(image):
    imageIn = np.asarray(image,np.uint8)
    rows = imageIn.shape[0]
    cols = imageIn.shape[1]

    # convert image to grayscale
    Y = [0.299,0.587,0.114]
    imGray = np.dot(imageIn,Y)
    imGray = imGray.astype(np.uint8)

    # convert ndarray into linear array
    linIm = np.reshape(imGray,rows * cols)

    # clip out of bounds values
    linIm = np.where(linIm < 0,0,linIm)
    linIm = np.where(linIm > 255,255,linIm)

    # reshape back into 2D array
    imGray = np.reshape(linIm,[rows,cols])

    return imGray

def histEQ(image):
    hist = np.zeros(256,dtype=int)
    freq = Counter(np.reshape(image,image.shape[0] * image.shape[1]))
    for p in range(256):
        hist[p] = freq[p]
    #plt.figure(dpi=170)
    #plt.plot(hist)
    #plt.title('Histogram of Intensities')
    #plt.ylabel('Number of pixels')
    #plt.xlabel('Intensity')
    remap = np.zeros(256,dtype=int)
    remap[-1] = 255
    histSum = sum(hist[1:-2])
    P = histSum / 254
    T = P
    outval = 1
    curr_sum = 0
    for inval in range(1,255,1):
        curr_sum += hist[inval]
        remap[inval] = outval
        if (curr_sum > T):
            outval = round(curr_sum/P)
            T = outval*P
    for intensity in range(256):
        imEQ = np.where(image == intensity, remap[intensity])
    return remap

im = Image.open('images/test01.jpg')
# convert image to grayscale
gray1 = grayscale(im)
im = Image.open('images/test02.jpg')
# convert image to grayscale
gray1 = grayscale(im)
# Output grayscale image to file
#Image.fromarray(gray1).save('figs/test01_out.jpg')
im = Image.open('images/lc1.jpg')
im = im.convert('L')
im = np.asarray(im,np.float)
im_equalized = histEQ(im)
im = Image.open('images/lc2.jpg')
im = im.convert('L')
im = np.asarray(im,np.float)
histogram = histEQ(im)