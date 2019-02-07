import numpy as np
from collections import Counter
from scipy import ndimage

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
    image = np.asarray(image,np.float)
    hist_before = np.zeros(256,dtype=int)
    freq = Counter(np.reshape(image,image.shape[0] * image.shape[1]))
    for p in range(256):
        hist_before[p] = freq[p]
    #plt.figure(dpi=170)
    #plt.plot(hist)
    #plt.title('Histogram of Intensities')
    #plt.ylabel('Number of pixels')
    #plt.xlabel('Intensity')
    remap = np.zeros(256,dtype=int)
    remap[-1] = 255
    histSum = sum(hist_before[1:-2])
    P = histSum / 254
    T = P
    outval = 1
    curr_sum = 0
    # build remap table
    for inval in range(1,255,1):
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
        image_equalized = np.where(image == intensity, remap[intensity], image_equalized)
    # compute histogram after equalization
    hist_after = np.zeros(256,dtype=int)
    freq = Counter(np.reshape(image_equalized,image_equalized.shape[0] * image_equalized.shape[1]))
    for p in range(256):
        hist_after[p] = freq[p]
    return image_equalized, hist_before, hist_after

def edgeDetect(imageIn):
    df_dy = np.array([-1, -2, -2],
                     [0, 0, 0]
                     [1, 2, 1])
    df_dx = np.array([-1, 0, 1]
                     [-2, 0, 2]
                     [-1, 0, 1])
    # perform convolution
    return imageIn