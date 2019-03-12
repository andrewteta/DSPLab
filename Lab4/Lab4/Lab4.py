import lab04_funcs as lf
import numpy as np
from tkinter import filedialog
import scipy.io.wavfile
import math
import matplotlib.pyplot as plt

c_taps = lf.load_c_taps('the_c_taps.txt')
d_taps = lf.load_d_taps('the_d_taps.txt')

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
# Loop over files selected
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]
    filepath = f
    # read .wav file into numpy array and extract sampling frequency
    fs,data = scipy.io.wavfile.read(filepath)
    # add zeros at end of array to make it reshapable into 32 cols
    data = np.append(data, np.zeros(32 - (len(data) % 32)))
    # reshape into 32 sample length rows
    data = np.reshape(data, (-1, 32))
    # initialize working arrays for subband coefficient calculation
    X = np.zeros(512)
    M = np.zeros([32, 64])
    # build filter matrix, M
    for k in range(32):
        for r in range(64):
            M[k,r] = math.cos(((2*k + 1) * (r - 16) * math.pi) / 64)
    # subband coefficient output matrix
    A = np.zeros_like(data)
    # helping vectors
    fInvert = np.array(16 * [1, -1])
    zFlat = np.array(8 * [1])

    # loop over entire song in 32 sample chunks
    for packet in range(np.shape(data)[0]):
        # shift every element of X to the right by 32
        X = np.roll(X, 32)
        # flip audio sample frame and place in X
        X[0:32] = np.flip(data[packet])
        # window X by C filter
        Z = c_taps * X
        # partial calculation
        Z = np.reshape(Z, (64, -1))
        Y = Z.dot(zFlat)
        # calculate 32 subband samples
        S = M.dot(Y)
        # undo frequency inversion and add to output array
        A[packet] = fInvert * S

    #coeffs = np.reshape(coeffs, (-1, 1))
    A = A.T.flatten()
    plt.figure()
    plt.plot(A)
    plt.show()
    


print('done')
