# Created by Andrew Teta
# 2019/02/13
# ECEN 4532 DSP Lab 3
# Perspective Transformations and Motion Tracking

import numpy as np
import matplotlib.pyplot as plt

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