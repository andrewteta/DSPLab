# Created by Andrew Teta
# 2019/02/13
# ECEN 4532 DSP Lab 3
# Perspective Transformations and Motion Tracking

import numpy as np
from PIL import Image
import lab3_funcs as lf

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

# calculate linear regression
linear_regression = lf.linReg(A, C)
print(f'x = {linear_regression[0]}\n')
print(f'MSE = {linear_regression[3]}\n')

# build perspective lines to be corrected
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

# build output lines
outputLine1 = np.zeros([10, 2])
outputLine2 = np.zeros([10, 2])
line1 = np.linspace(750, 405, 10)
line2 = np.linspace(750, 405, 10)
for i in range(10):
    outputLine1[i] = [line1[i], 305]
    outputLine2[i] = [line2[i], 475]

# combine output lines into one vector of points
outputPoints = np.append(outputLine1, outputLine2, axis=0)
print(f'outputPoints = {outputPoints}\n shape(outputPoints) = {np.shape(outputPoints)}\n')

# correct distortion
distImage = Image.open('PC_test_2.jpg')
distImage = np.asarray(distImage)
perspective_correct = lf.distortion_correction(distImage, inputPoints, outputPoints)

# display and save image
Image.fromarray(perspective_correct[0].astype(np.uint8)).show()
Image.fromarray(perspective_correct[0].astype(np.uint8)).save('out1.jpg')
print('done')

line = np.linspace(1080, 0, 10)
y1 = 215
y2 = 575
# build output lines
outputLine1 = np.zeros([10, 2])
outputLine2 = np.zeros([10, 2])
for i in range(10):
    outputLine1[i] = [line[i], y1]
    outputLine2[i] = [line[i], y2]
# combine output lines into one vector of points
outputPoints1 = np.append(outputLine1, outputLine2, axis=0)

# correct distortion
perspective_correct1 = lf.distortion_correction(distImage, inputPoints, outputPoints1)
# display and save image
Image.fromarray(perspective_correct1[0].astype(np.uint8)).show()
Image.fromarray(perspective_correct1[0].astype(np.uint8)).save('out2.jpg')