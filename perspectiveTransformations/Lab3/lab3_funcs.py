import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import bilinear_interp as interp
import me_method as me

def linReg(A, C):
    # compute pseudo inverse matrix
    A_p = np.linalg.inv(A.T.dot(A)).dot(A.T)
    #print(f'A_p = {A_p}\n')
    # compute weighting coefficient vector
    x = A_p.dot(C)
    #print(f'x = {x}\n')
    # compute mapped values
    v = A.dot(x)
    #print(f'v = {v}\n')
    # compute mean-squared-error of solutions
    mse = (abs(v-C))**2
    #print(f'MSE = {mse}')
    
    return x, v, A_p, mse

def distortion_correction(imageIn, inPoints, outPoints):
    # build remapping matrix for calculation of H
    mapping = np.zeros([0, 8])
    for row in range(20):
        ip1 = inPoints[row, 0]
        ip2 = inPoints[row, 1]
        op1 = outPoints[row, 0]
        op2 = outPoints[row, 1]
        arr = np.asarray([[ip1, ip2, 1, 0, 0, 0, -op1*ip1, -op1*ip2],
                          [0, 0, 0, ip1, ip2, 1, -op2*ip1, -op2*ip2]])
        mapping = np.append(mapping, arr, axis=0)
    #print(f'mapping = {mapping}\n shape(mapping) = {np.shape(mapping)}\n')

    # calculate pseudo inverse to find H
    mapping_inv = np.linalg.inv(mapping.T.dot(mapping)).dot(mapping.T)
    outPoints = np.reshape(outPoints, [40, 1])
    #print(f'outPoints = {outPoints}\n')

    # find H by taking the dot product of H* and output points
    H = mapping_inv.dot(outPoints)
    #print(f'shape(H) = {np.shape(H)}')
    H = np.append(H, 1)
    H = np.reshape(H, [3,3])
    #print(f'H = {H}\n')
    H_inv = np.linalg.inv(H)

    # correct distortion of perspective image
    corrImage = np.zeros([np.shape(imageIn)[0], np.shape(imageIn)[1], 3])
    print('remapping pixels...')
    # take the dot product of H with each pixel in distorted image
    for y in range(np.shape(corrImage)[0]):
        for x in range(np.shape(corrImage)[1]):
            p = [x, y, 1]
            v = H_inv.dot(p)
            # calculate points in distorted image corresponding to corrected pixels
            map = [v[0]/v[2], v[1]/v[2]]
            # interpolate to find real pixel values
            corrImage[y, x, :] = interp.bilinear_interp(map[0], map[1], imageIn)
        # print progress bar
        progress = int(y*100/np.shape(corrImage)[0])
        sys.stdout.write('{0}% complete...\r'.format(progress))
        sys.stdout.flush()
    print('\n')

    return corrImage, H, imageIn

def detect_motion(frame1, frame2, blockSize):
    im1 = np.asarray(Image.open(frame1))
    im2 = np.asarray(Image.open(frame2))
    im1_bw = np.asarray(Image.open(frame1).convert('L'), np.float)
    im2_bw = np.asarray(Image.open(frame2).convert('L'), np.float)

    rows = np.shape(im2)[0]
    cols = np.shape(im2)[1]
    blockSize = 16

    predicted_image = np.zeros([rows, cols, 3], np.uint8)
    MSE = np.zeros([rows, cols], np.float)
    vectorCount = int((rows/blockSize) * (cols/blockSize))
    U = np.zeros(vectorCount)
    V = np.zeros(vectorCount)
    X = np.zeros(vectorCount)
    Y = np.zeros(vectorCount)
    currIndex = 0
    # loop over blocks
    print('Estimating motion ... \n')
    for row in range(0, rows, blockSize):
        for col in range(0, cols, blockSize):
            block = im2_bw[row:row + blockSize, col:col + blockSize]
            motion_estimation = me.motion_match(row, col, 20, block, im1_bw, im1)
            predicted_image[row:row + blockSize, col:col + blockSize] = motion_estimation[0]
            MSE[row, col] = motion_estimation[1]
            U[currIndex] = motion_estimation[3]
            V[currIndex] = -motion_estimation[2]
            X[currIndex] = int(col/blockSize)
            Y[currIndex] = rows - int(row/blockSize)
            currIndex += 1
        progress = int((row*100)/rows)
        sys.stdout.write('{0}% complete...\r'.format(progress))
        sys.stdout.flush()
    print('\n')
    Image.fromarray(predicted_image.astype(np.uint8)).save('predicted.jpg')
    plt.figure(dpi=170)
    plt.quiver(X, Y, U, V)
    plt.title("Motion Estimation")
    plt.xlabel("Horizontal Block Number")
    plt.ylabel("Vertical Block Number")
    plt.ion()
    plt.savefig('vector_field.png')

    return 0

def detect_motion_mae(frame1, frame2, blockSize):
    im1 = np.asarray(Image.open(frame1))
    im2 = np.asarray(Image.open(frame2))
    im1_bw = np.asarray(Image.open(frame1).convert('L'), np.float)
    im2_bw = np.asarray(Image.open(frame2).convert('L'), np.float)

    rows = np.shape(im2)[0]
    cols = np.shape(im2)[1]
    blockSize = 16

    predicted_image = np.zeros([rows, cols, 3], np.uint8)
    MSE = np.zeros([rows, cols], np.float)
    vectorCount = int((rows/blockSize) * (cols/blockSize))
    U = np.zeros(vectorCount)
    V = np.zeros(vectorCount)
    X = np.zeros(vectorCount)
    Y = np.zeros(vectorCount)
    currIndex = 0
    # loop over blocks
    print('Estimating motion ... \n')
    for row in range(0, rows, blockSize):
        for col in range(0, cols, blockSize):
            block = im2_bw[row:row + blockSize, col:col + blockSize]
            motion_estimation = me.motion_match_mae(row, col, 20, block, im1_bw, im1)
            predicted_image[row:row + blockSize, col:col + blockSize] = motion_estimation[0]
            MSE[row, col] = motion_estimation[1]
            U[currIndex] = motion_estimation[3]
            V[currIndex] = -motion_estimation[2]
            X[currIndex] = int(col/blockSize)
            Y[currIndex] = rows - int(row/blockSize)
            currIndex += 1
        progress = int((row*100)/rows)
        sys.stdout.write('{0}% complete...\r'.format(progress))
        sys.stdout.flush()
    print('\n')
    Image.fromarray(predicted_image.astype(np.uint8)).save('predicted1.jpg')
    plt.figure(dpi=170)
    plt.quiver(X, Y, U, V)
    plt.title("Motion Estimation")
    plt.xlabel("Horizontal Block Number")
    plt.ylabel("Vertical Block Number")
    plt.ion()
    plt.savefig('vector_field1.png')

    return 0