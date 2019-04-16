import numpy as np
import math
import scipy.fftpack
from PIL import Image

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
[6, 1],
[7, 0],
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
    #print(f'before = \n{A}\n')
    X = scipy.fftpack.dct(A, axis = 0, norm='ortho')
    Y = scipy.fftpack.dct(X, axis = 1, norm='ortho')
    #print(f'dct = \n{Y}\n')
    return Y

def idct_2D(A):
    Y = scipy.fftpack.idct(A, axis = 0, norm='ortho')
    X = scipy.fftpack.idct(Y, axis = 1, norm='ortho')
    #print(f'after = \n{X}\n')
    return X

def zigzag(A):
    #print(f'before zigzag = \n{A}\n\n')
    Y = np.zeros(64)
    for n in range(len(zz)):
        map = zz[n]
        Y[n] = A[map[0], map[1]]
    #print(f'after zigzag = \n{Y}\n\n')
    return Y

def izigzag(A):
    #print(f'before inverse zigzag = \n{A}\n\n')
    X = np.zeros((8, 8))
    for n in range(len(A)):
        map = zz[n]
        X[map[0], map[1]] = A[n]
    #print(f'after inverse zigzag = \n{X}\n\n')
    return X

def dctmgr(image, loss_factor):
    processed_image = np.zeros_like(image)
    dct_array = np.zeros((64, int(np.shape(image)[0] * np.shape(image)[1] / 64)))
    blockSize = 8
    Ny, Nx = np.shape(image)
    nBlocksX = int(Nx/blockSize)
    nBlocksY = int(Ny/blockSize)
    block = 0
    #print(f'block before processing = \n{image[0:8,0:8]}\n')
    #print(f'block after 2d dct = \n{dct_2D(image[0:8,0:8])}\n')
    # loop over 8x8 block cols
    for row in range(nBlocksX):
        for col in range(nBlocksY):
            indexX = col * 8
            indexY = row * 8
            #print(f'indexX (col) = {indexX}')
            #print(f'indexY (row) = {indexY}')
            # slice out an 8x8 block
            pix = image[indexY : indexY + blockSize, indexX : indexX + blockSize]
            #print(f'input = \n{pix}\n')
            # take 2D DCT transform
            dct_pix = dct_2D(pix)
            #print(f'dct = \n{dct_pix}\n')
            # quantize
            q = quant_coeffs(dct_pix, loss_factor)
            #print(f'q = \n{q}\n')
            # re-order block in zigzag pattern and place in output array
            zz = zigzag(q)
            dct_array[:, block] = zz
            #print(f'zz = \n{zz}\n')
            #print(f'dct_array[i] = \n{dct_array[:, block]}\n')
            #iz = izigzag(zz)
            #print(f'izz = \n{iz}\n')
            #iq = iquant_coeffs(iz, 1)
            #print(f'iq = \n{iq}\n')
            #y = idct_2D(iq)
            #print(f'idct = \n{y}\n')
            #print(f'block = {block}')
            block += 1
    # quantize values according to Q and loss factor
    #output = quant_coeffs(dct_array, loss_factor)
    output = dct_array
    #print(f'first column after compression = \n{output[:,0]}\n')
    return output

def idctmgr(input, loss_factor):
    nPix = int(np.shape(input)[1] / 8)
    output = np.zeros((nPix, nPix))
    #coeffs = input
    for col in range(np.shape(input)[1]):
        #print(f'input col = \n{input[:,col]}\n')
        indexX = (col % 64) * 8
        indexY = (int(col / 64)) * 8
        #print(f'index (row={indexX},col={indexY})\n')
        zag = izigzag(input[:, col])
        # invert quantization
        coeffs = iquant_coeffs(zag, loss_factor)
        #print(f'izz = \n{zag}\n')
        idct = idct_2D(coeffs)
        #print(f'idct = \n{idct}\n')
        output[indexY:indexY + 8, indexX:indexX + 8] = idct
        #Image.fromarray(output).show()
        #print('\n')
    return output

def quant_coeffs(A, loss_factor):
    #print(f'before quant = \n{A[0:8,0:8]}\n')
    #A[1:,:] = np.floor((np.divide(A[1:,:], zigzag(Q)[1:, np.newaxis]) / loss_factor) + 0.5)
    z = A[0,0]
    A = np.floor((A / (loss_factor * Q)) + 0.5)
    A[0,0] = z
    #print(f'after quant = \n{A[0:8,0:8]}\n')
    return A

def iquant_coeffs(A, loss_factor):
    #A[1:,:] = A[1:,:] * loss_factor * zigzag(Q)[1:, np.newaxis
    z = A[0,0]
    A = A * loss_factor * Q
    A[0,0] = z
    #print(f'after iquant = \n{A[0:8,0:8]}\n')
    return A

def enc_rbv(A):
    symb = np.zeros( (0,3), np.int )
    for i in range(np.shape(A)[1]):
        # Handle DC coeff
        symb = np.vstack( [symb, [0, 12, A[0,i]]] )

        # Handle AC coeffs. nzi means non-zero indices
        tmp = A[:,i].flatten()
        nzi = np.where( tmp[1:] != 0 ) [0]
        prev_index = 0
        for k in range(len(nzi)):
            curr_index = nzi[k] + 1
            zeros = curr_index - prev_index - 1
            symb = np.vstack([symb, [0,0,0]])
        return symb
    return Y

def dec_rbv(A):
    return X