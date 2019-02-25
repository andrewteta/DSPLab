import numpy as np
import bilinear_interp as interp
import sys

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

    return corrImage, H, imageIn
