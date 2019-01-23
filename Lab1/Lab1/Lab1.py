# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:24:09 2019

@author: Andrew Teta
"""

import scipy.io.wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

# =============================================================================
# filepath = string containing absolute path to audio file
# duration = length of sample to extract in seconds
# location = start index of sample given in seconds
# =============================================================================
def extractSample(filepath, duration, location):
    fs,data = scipy.io.wavfile.read(filepath)
    print('Read file: '+filepath+'\n')
    Tsample = 1/fs
    startIndex = int(location/Tsample)
    endIndex = startIndex + int(duration/Tsample)
    return fs,data[startIndex:endIndex]

# =============================================================================
# sample = audio clip
# N = frame size
# =============================================================================
def loudness(sample,N):
    nframes = int(len(sample)/N)
    E = np.zeros(nframes)
    sigma = np.zeros(nframes)
    for n in range(nframes):
        E[n] = (1/N)*sum(sample[n*N:n*N+(N-1)])
    for n in range(nframes):
        sigma[n] = np.sqrt((1/N)*sum((sample[n*N:n*N+(N-1)] - E[n])**2))
    return E,sigma,nframes

def zcr(sample,N):
    nframes = int(len(sample)/N)
    Z = np.zeros(nframes)
    for n in range(nframes):
        Z[n] = 1/(N-1) * sum(0.5*abs(np.sign(sample[n*N+1:n*N+(N-1)]) - np.sign(sample[n*N:n*N+(N-2)])))
    return Z, nframes

files = filedialog.askopenfilenames()
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]

    filepath = f
    fs,songSample = extractSample(filepath,24,120)
    E,sigma,Nloudness = loudness(songSample,512)
    Z,Nzcr = zcr(songSample,512)

    plt.figure()
    plt.plot(range(Nloudness),sigma)
    plt.xlabel('frames')
    plt.ylabel('loudness')
    plt.title('Loudness: '+filename)
    plt.savefig('figs/loudness_'+filename)
    plt.close()

    plt.figure()
    plt.plot(range(Nzcr),Z)
    plt.xlabel('frames')
    plt.ylabel('ZCR')
    plt.title('ZCR: '+filename)
    plt.savefig('figs/zcr_'+filename)
    plt.close()

w = scipy.signal.get_window('hann',512)
plt.figure()
plt.plot(w)
plt.title('Hann Window')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.savefig('figs/hann')
plt.close()
w = scipy.signal.get_window('blackman',512)
plt.figure()
plt.plot(w)
plt.title('Blackman Window')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
plt.savefig('figs/blackman')
plt.close()

def FFT(sample,fs,N):
    nframes = int(len(sample)/N)
    w = scipy.signal.get_window('blackman',N-1)
    for n in range(nframes):
        s = sample[n*N:n*N+(N-1)]
        f,t,s = scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,w)
        plt.figure()
        plt.plot(f,s)
        plt.show()

FFT(songSample,fs,512)
    



