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
# Input:
#   filepath = string containing absolute path to audio file
#   duration = length of sample to extract in seconds
#   location = start index of sample given in seconds
# Output:
#   arg1 = sampling frequency (Hz)
#   arg2 = sample of specified duration in seconds
# =============================================================================
def extractSample(filepath, duration, location):
    fs,data = scipy.io.wavfile.read(filepath)
    print('Read file: '+filepath+'\n')
    Tsample = 1/fs
    startIndex = int(location/Tsample)
    endIndex = startIndex + int(duration/Tsample)
    return fs,data[startIndex:endIndex]

# =============================================================================
# Input:
#   sample = audio clip
#   N = frame size
# Output:
#   arg1 = energy vector
#   arg2 = standard deviation vector
#   arg3 = num frames (for accurate plotting)
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

# =============================================================================
# Input:
#   sample = audio clip
#   N = frame size
# Output:
#   arg1 = zero-crossing-rate
#   arg2 = num frames (for accurate plotting)
# =============================================================================
def zcr(sample,N):
    nframes = int(len(sample)/N)
    Z = np.zeros(nframes)
    for n in range(nframes):
        Z[n] = 1/(N-1) * sum(0.5*abs(np.sign(sample[n*N+1:n*N+(N-1)]) - np.sign(sample[n*N:n*N+(N-2)])))
    return Z, nframes

# =============================================================================
# Input:
#   sample = audio clip
#   fs = sampling frequency
#   N = frame size
# Output:
#   arg1 = spectrogram using Hann window
#   arg2 = spectrogram using Blackman window
#   arg3 = vector of axis points in frequency (Hz)
#   arg4 = vector of sxis points in time (s)
# =============================================================================
def spectrogram(sample,fs,N):
    # generate a Hann window
    wHann = scipy.signal.get_window('hann',N-1,False)
    # Save figure
    plt.figure(dpi=170)
    plt.plot(wHann)
    plt.title('Hann Window')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.savefig('figs/hann')
    plt.close()
    # Generate Blackman window
    wBlack = scipy.signal.get_window('blackman',N-1,False)
    # save figure
    plt.figure(dpi=170)
    plt.plot(wBlack)
    plt.title('Blackman Window')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.savefig('figs/blackman')
    plt.close()
    # Number of FFT frames
    nframes = int(len(sample)/N)
    SHann = np.zeros([nframes,int(N/2)])
    SBlack = np.zeros([nframes,int(N/2)])
    time = np.zeros(nframes)
    # Loop over FFT frames
    for n in range(nframes):
        # Calculate a frequency power spectrum for Hann window
        f,t,sH = np.transpose(scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,wHann,mode='magnitude'))
        # throw away values smaller than 10^-3
        sH = np.where(sH < 10**-3, 10**-3, sH)
        # convert linear values to dB scale
        sH = 20*np.log10(sH/np.amax(sH))
        # Repeat for Blackman window
        f,t,sB = np.transpose(scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,wBlack,mode='magnitude'))
        sB = np.where(sB < 10**-3, 10**-3, sB)
        sB = 20*np.log10(sB/np.amax(sB))
        # Reshape a bit
        SHann[n,:] = np.transpose(sH)
        SBlack[n,:] = np.transpose(sB)
        time[n] = t
    return SHann,SBlack,f,time

def spectralCentroid(sp,N):
    # THESE ARE IN DECIBELS...
    nframes = np.shape(sp)[1]
    frange = np.shape(sp)[0]
    P = np.zeros([nframes,frange])
    # Loop over FFT frames
    for n in range(nframes):
        # calculate sum of all frequency components in this frame
        sfreq = sum(sp[0:frange,n])
        # for each frequency bin, calculate the relative 'probability' of that frequency
        for k in range(frange):
            P[n,k] = sp[k,n]/sfreq
    return P,N

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
# Loop over files selected
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]

    filepath = f
    # Extract 24s sample from center of 60s clip
    fs,songSample = extractSample(filepath,24,60)
    # Calculate loudness (energy function returned just in case)
    E,sigma,Nloudness = loudness(songSample,512)
    # Calculate zero-crossing-rate
    Z,Nzcr = zcr(songSample,512)

    # Save figures
    plt.figure(dpi=170)
    plt.plot(range(Nloudness),sigma)
    plt.xlabel('frames')
    plt.ylabel('loudness')
    plt.title('Loudness: '+filename)
    plt.savefig('figs/loudness_'+filename)
    plt.close()

    plt.figure(dpi=170)
    plt.plot(range(Nzcr),Z)
    plt.xlabel('frames')
    plt.ylabel('ZCR')
    plt.title('ZCR: '+filename)
    plt.savefig('figs/zcr_'+filename)
    plt.close()

    # Calculate spectrograms for Hann and Blackman windows
    spectroHann,spectroBlack,freqAxis,timeAxis = spectrogram(songSample,fs,512)
    spectroHann = np.transpose(spectroHann)
    spectroBlack = np.transpose(spectroBlack)

    # save figures
    plt.figure(dpi=170)
    plt.imshow(spectroHann, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar
    plt.title('Hann Window Spectrogram: '+filename)
    plt.xlabel('FFT Frame')
    plt.ylabel('Frequency (bins)')
    plt.savefig('figs/powerBlack_'+filename)
    plt.close()

    plt.figure(dpi=170)
    plt.imshow(spectroBlack, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar
    plt.title('Blackman Window Spectrogram: '+filename)
    plt.xlabel('FFT Frame')
    plt.ylabel('Frequency (bins)')
    plt.savefig('figs/powerHann_'+filename)
    plt.close()

    dft = spectralCentroid(spectroBlack,512)

print('done')
    



