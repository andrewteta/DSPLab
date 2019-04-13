import numpy as np
import math
import scipy.fftpack

zz = np.asarray([
[0, 0],
[0, 1],
[1, 0],
[2, 0],
[1, 1],
[0, 2],
[0, 3],
[1, 2],
[2, 1],
[3, 0],
[4, 0],
[3, 1],
[2, 2],
[1, 3],
[0, 4],
[0, 5],
[1, 4],
[2, 3],
[3, 2],
[4, 1],
[5, 0],
[6, 0],
[5, 1],
[4, 2],
[3, 3],
[2, 4],
[1, 5],
[0, 6],
[0, 7],
[1, 6],
[2, 5],
[3, 4],
[4, 3],
[5, 2],
[5, 1],
[6, 0],
[7, 1],
[6, 2],
[5, 3],
[4, 4],
[3, 5],
[2, 6],
[1, 7],
[2, 7],
[3, 6],
[4, 5],
[5, 4],
[6, 3],
[7, 2],
[7, 3],
[6, 4],
[5, 5],
[4, 6],
[3, 7],
[4, 7],
[5, 6],
[6, 5],
[7, 4],
[7, 5],
[6, 6],
[5, 7],
[6, 7],
[7, 6],
[7, 7] ], np.int)

Q = np.asarray([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99] ], np.float)

def dct_2D(A):
    X = scipy.fftpack.dct(A, axis = 0, norm='ortho')
    Y = scipy.fftpack.dct(X, axis = 1, norm='ortho')
    return Y

def idct_2D(A):
    Y = scipy.fftpack.idct(A, axis = 0, norm='ortho')
    X = scipy.fftpack.idct(Y, axis = 1, norm='ortho')
    return X

def zigzag(A):
    print(f'before zigzag = \n{A}\n\n')
    Y = np.zeros(64)
    for n in range(len(zz)):
        map = zz[n]
        Y[n] = A[map[0], map[1]]
    print(f'after zigzag = \n{Y}\n\n')
    return Y

def izigzag(A):
    return X

def dctmgr(image):
    processed_image = np.zeros_like(image)
    dct_array = np.zeros((64, (int)(np.shape(image)[0] * np.shape(image)[1] / 64)))
    blockSize = 8
    Ny, Nx = np.shape(image)
    nBlocksX = int(Nx/blockSize)
    nBlocksY = int(Ny/blockSize)
    # loop over 8x8 block cols
    for blockX in range(nBlocksX):
        for blockY in range(nBlocksY):
            # slice out an 8x8 block
            pix = image[blockY : blockY + blockSize, blockX : blockX + blockSize]
            # take 2D DCT transform
            dct_pix = dct_2D(pix)
            zig = zigzag(pix)
            # calculate and place block in zig-zag pattern location
    return processed_image

# rows * cols / 64 = numblocks
# make this an array with 64 rows


def quant_coeffs(A):
    return Y

def iquant_coeffs(A):
    return X

def enc_rbv(A):
    return Y

def dec_rbv(A):
    return X