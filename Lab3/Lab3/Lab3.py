# Created by Andrew Teta
# 2019/02/13
# ECEN 4532 DSP Lab 3
# Perspective Transformations and Motion Tracking

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

A = np.asarray([[1,8,3,65.66],
                [-46,-98,108,-1763.1],
                [5,12,-9,195.2],
                [63,345,-27,3625],
                [23,78,45,716.9],
                [-12,56,-8,339],
                [1,34,78,-25.5],
                [56,123,-5,1677.1]])
C = A[:,3]
A = A[:,0:-1]
print(f'A = {A}\n')
print(f'C = {C}\n')
# compute pseudo inverse matrix
A_p = np.linalg.inv(A.T.dot(A)).dot(A.T)
print(f'A_p = {A_p}\n')
# compute weighting coefficient vector
x = A_p.dot(C)
print(f'x = {x}\n')
# compute mapped values
v = A.dot(x)
print(f'v = {v}\n')
# compute mean-squared-error of solutions
mse = (abs(v-C))**2
print(f'MSE = {mse}')

inputPoints = np.asarray([[750, 215],
                          [675, 235],
                          [625, 250],
                          [575, 265],
                          [535, 275],
                          [500, 285],
                          [475, 295],
                          [445, 300],
                          [425, 305],
                          [405, 315],
                          [745, 575],
                          [670, 555],
                          [615, 540],
                          [565, 525],
                          [530, 515],
                          [495, 505],
                          [468, 500],
                          [440, 495],
                          [420, 488],
                          [400, 483]])
print(f'inputPoints = {inputPoints}\n')

outputLine1 = np.zeros([10, 2])
outputLine2 = np.zeros([10, 2])
line1 = np.linspace(751, 405, 10)
line2 = np.linspace(751, 405, 10)

for i in range(10):
    outputLine1[i] = [line1[i], 305]
    outputLine2[i] = [line2[i], 475]

outputPoints = np.append(outputLine1, outputLine2, axis=0)
print(f'outputPoints = {outputPoints}\n shape(outputPoints) = {np.shape(outputPoints)}\n')
# calculate pseudo inverse

mapping = np.zeros([0, 8])
for row in range(20):
    ip1 = inputPoints[row, 0]
    ip2 = inputPoints[row, 1]
    op1 = outputPoints[row, 0]
    op2 = outputPoints[row, 1]
    arr = np.asarray([[ip1, ip2, 1, 1, 1, 1, -op1*ip1, -op1*ip2],
                      [0, 0, 0, ip1, ip2, 1, -op2*ip1, -op2*ip2]])
    mapping = np.append(mapping, arr, axis=0)

print(f'mapping = {mapping}\n shape(mapping) = {np.shape(mapping)}\n')

mapping_inv = np.linalg.inv(mapping.T.dot(mapping)).dot(mapping.T)
outputPoints = np.reshape(outputPoints, [40, 1])
H = mapping_inv.dot(outputPoints)
print(f'shape(H) = {np.shape(H)}')
H = np.append(H, 1)
H = np.reshape(H, [3,3])
print(f'H = {H}\n')

distImage = Image.open('PC_test_2.jpg')
distImage = np.asarray(distImage.convert('L'), np.float)
corrImage = np.zeros_like(distImage)

for y in range(np.shape(corrImage)[0]):
    for x in range(np.shape(corrImage)[1]):
        p = [x, y, 1]
        v = H.dot(p)
        map = [v[0]/v[2], v[1]/v[2]]
        print(f'map = {map}\n')
        if (map[0] > np.shape(distImage)[0] or map[1] > np.shape(distImage)[1]):
            break
