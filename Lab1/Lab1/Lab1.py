# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:24:09 2019

@author: Andrew Teta
"""

import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

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
    return E,sigma

filepath = 'audio/track201-classical.wav'
fs,songSample = exSample(filepath,24,120)
E,sigma = loudness(songSample,512)
plt.figure()
plt.plot(sigma)
plt.show()




