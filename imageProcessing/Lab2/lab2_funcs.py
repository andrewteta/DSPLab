import numpy as np
from collections import Counter
from scipy import signal
from scipy.interpolate import RectBivariateSpline as sp
from PIL import Image

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
    image = image.convert('L')
    image = np.asarray(image, np.float)
    hist_before = np.zeros(256, dtype=int)
    # Count frequency of every value in image
    freq = Counter(np.reshape(image, image.shape[0] * image.shape[1]))
    # sort into numpy array
    for p in range(256):
        hist_before[p] = freq[p]
    # declare remapping table and initialize variables
    remap = np.zeros(256, dtype=int)
    remap[-1] = 255
    histSum = sum(hist_before[1:-2])
    P = histSum / 254
    T = P
    outval = 1
    curr_sum = 0
    # build remap table
    for inval in range(1, 255, 1):
        curr_sum += hist_before[inval]
        remap[inval] = outval
        if (curr_sum > T):
            outval = round(curr_sum/P)
            T = outval*P
    # declare output image
    image_equalized = np.zeros_like(image)
    # remap intensities into equalized array
    for intensity in range(256):
        # remap values to equalize
        image_equalized = np.where(image == intensity, 
                                   remap[intensity], 
                                   image_equalized)
    # compute histogram after equalization
    hist_after = np.zeros(256,dtype=int)
    # count value occurrences
    freq = Counter(np.reshape(image_equalized,image_equalized.shape[0] 
                              * image_equalized.shape[1]))
    # sort
    for p in range(256):
        hist_after[p] = freq[p]
    return image_equalized, hist_before, hist_after, remap, image

def Sobel(imageIn, thresh):
    # convert to grayscale
    imageIn = np.asarray(imageIn.convert('L'), np.float)
    convolved = np.zeros_like(imageIn)
    # Sobel kernels
    df_dy = np.array([[-1, -2, -2],
                     [0, 0, 0],
                     [1, 2, 1]])
    df_dx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
    # perform convolution
    convX = signal.fftconvolve(imageIn, df_dx, mode='same')
    convY = signal.fftconvolve(imageIn, df_dy, mode='same')
    # find the magnitude of the gradient for every pixel
    gradient = np.sqrt( (convX**2) + (convY**2) )
    # normalize
    gradient = (gradient/np.amax(gradient)) * 255
    # detect edges based on threshold value
    imageOut = np.where(gradient < thresh, 0, 255)
    return imageOut

def scaleDown(imageIn, N):
    # convert to grayscale
    imageIn = np.asarray(imageIn.convert('L'), np.float)
    # declare new, smaller image
    imageOut = np.zeros([(int)(np.shape(imageIn)[0] / N), 
                         (int)(np.shape(imageIn)[1] / N)])
    xIndex = 0
    yIndex = 0
    # loop over x dim
    for blockX in range(0, np.shape(imageIn)[0], N):
        # loop over y dim
        for blockY in range(0, np.shape(imageIn)[1], N):
            sample = imageIn[blockX:blockX + N, blockY:blockY + N]
            imageOut[xIndex,yIndex] = np.average(sample)
            yIndex += 1
        yIndex = 0
        xIndex += 1
    return imageOut

def upScale(imageIn, N):
    # convert to grayscale
    imageIn = np.asarray(imageIn.convert('L'), np.float)
    # declare new, larger image
    imageOut = np.zeros([(int)(np.shape(imageIn)[0] * N), 
                         (int)(np.shape(imageIn)[1] * N)])
    xIndex = 0
    yIndex = 0
    # loop over every x pixel in input image
    for pixelX in range(np.shape(imageIn)[0] - 1):
        # loop over every y pixel in input image
        for pixelY in range(np.shape(imageIn)[1] - 1):
            # bi-linear interpolation
            # x and y hold indices of pixels used for interpolation
            x = np.asarray([pixelX,pixelX + 1], np.int)
            y = np.asarray([pixelY,pixelY + 1], np.int)
            # z holds intensities at those locations
            z = np.asarray([[imageIn[pixelX,pixelY],
                             imageIn[pixelX,pixelY + 1]],
                            [imageIn[pixelX + 1,pixelY],
                             imageIn[pixelX + 1,pixelY + 1]]], np.float)
            # interpolation object
            interp_spline = sp(y, x, z, kx=1, ky=1)
            # interpolate onto smaller grid
            x1 = np.linspace(pixelX, pixelX + 1, N + 1)
            y1 = np.linspace(pixelY, pixelY + 1, N + 1)
            ivals = interp_spline(y1, x1)
            # spread pixels out and place interpolated values in between
            imageOut[pixelX*N:(pixelX*N) + (N), pixelY*N:(pixelY*N) + (N)] = ivals[0:N,0:N]
    return imageOut

def difference(im1, im2):
    diff = 2*np.abs(im1 - im2)
    np.where(diff > 255, 255, diff)
    return diff