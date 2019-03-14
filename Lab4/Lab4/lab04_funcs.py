#!/usr/local/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile

def load_c_taps( filename_path ):
  fd = open(filename_path, 'r')
  c = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    c[ctr] = np.float( value )
  fd.close()
  return c

def load_d_taps( filename_path ):
  fd = open(filename_path, 'r')
  d = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    d[ctr] = np.float( value )
  fd.close()
  return d

def pqmf(input):
    filename = str.split(input,'/')[-1]
    filename = str.split(filename,'.')[0]
    filepath = input

    # read .wav file into numpy array and extract sampling frequency
    inFile = scipy.io.wavfile.read(filepath)
    print('Analyzing ' + filename + ' ... \n')

    # load window coefficients from file
    c_taps = load_c_taps('the_c_taps.txt')

    # extract sample data and sampling frequency
    data = inFile[1]
    fs = inFile[0]

    # extract first 5 seconds of .wav file
    data = data[0 : int(5 * fs)]

    plt.figure(dpi=170)
    plt.plot(data, linewidth=0.25)

    # add zeros at end of array to make it reshapable into 32 cols
    data = np.append(data, np.zeros(32 - (len(data) % 32)))
    # reshape into 32 sample length rows
    data = np.reshape(data, (-1, 32))

    # initialize working arrays for subband coefficient calculation
    X = np.zeros(512)

    # build filter matrix, M
    M = np.zeros([32, 64])
    for k in range(32):
        for r in range(64):
            M[k,r] = math.cos((( (2 * k) + 1) * (r - 16) * math.pi) / 64)

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
        Z = np.reshape(Z, (8, 64))
        Y = zFlat.dot(Z)
        # calculate 32 subband samples
        S = M.dot(Y)
        # undo frequency inversion and add to output array
        A[packet, :] = fInvert * S

    # transpose and flatten to sort data into incrementing groups of subband coefficients
    A = A.T.flatten()
    # normalize
    A = A / np.amax(A)

    # plot sub-band coefficients
    plt.figure(dpi=170)
    plt.plot(A, linewidth=0.25)
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.title('Sub-band coefficients for first 5 seconds of ' + filename + '.wav')
    #plt.show()
    plt.savefig('./figures/' + filename + '_sb_filt.png')
    #plt.close()

    return A

def ipqmf(input):
    # add zeros at end of array to make it reshapable into 32 cols
    data = np.append(input, np.zeros(32 - (len(input) % 32)))
    # reshape into 32 sample length rows
    data = np.reshape(data, (-1, 32))

    # load window coefficients from file
    d_taps = load_d_taps('the_d_taps.txt')

    # declare working array
    V = np.zeros(1024)
    U = np.zeros(512)
    W = np.zeros(512)

    # reconstruction coefficient output matrix
    S = np.zeros_like(data)

    # build reconstruction matrix, N
    N = np.zeros([64, 32])
    for i in range(64):
        for k in range(32):
            N[i,k] = math.cos((( (2 * k) + 1) * (16 + i) * math.pi) / 64)

    # helping vectors
    fInvert = np.array(16 * [1, -1])
    wFlat = np.array(16 * [1])

    # loop over coefficients in 32 sample chunks
    for packet in range(np.shape(data)[0]):
        # shift every element of V to the right by 64
        V = np.roll(V, 64)
        # compute reconstruction samples
        V[0:64] = N.dot(fInvert * data[packet])
        # build window operand
        for i in range(8):
            for j in range(32):
                U[i * 64 + j] = V[i * 128 + j]
                U[i * 64 + 32 + j] = V[i * 128 + 96 + j]
        # window
        W = U * d_taps
        W = np.reshape(W, (16, -1))
        S[packet, :] = wFlat.dot(W)

    # transpose and flatten to sort data into incrementing groups of subband coefficients
    S = S.T.flatten()
    # normalize
    S = S / np.amax(S)

    return S