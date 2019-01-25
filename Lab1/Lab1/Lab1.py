# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:24:09 2019

@author: Andrew Teta
"""

import scipy.io.wavfile
import scipy.signal
import scipy.stats
import numpy as np
import math
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
    # Generate Blackman window
    wBlack = scipy.signal.get_window('blackman',N-1,False)
    # Save figure
    plt.figure(dpi=170)
    plt.plot(wHann)
    plt.title('Hann Window')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.savefig('figs/hann')
    plt.close()
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
        # throw away values smaller than 10^-3 and square values
        sH = sH**2
        sH = np.where(sH < 10**-3, 10**-3, sH)
        # Repeat for Blackman window
        f,t,sB = np.transpose(scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,wBlack,mode='magnitude'))
        sB = sB**2
        sB = np.where(sB < 10**-3, 10**-3, sB)
        # Reshape a bit
        SHann[n,:] = np.transpose(sH*fs/N)
        SBlack[n,:] = np.transpose(sB*fs/N)
        time[n] = t
    # convert linear values to dB scale
    SHann = 20*np.log10(SHann/np.amax(SHann))
    SBlack = 20*np.log10(SBlack/np.amax(SBlack))
    return SHann,SBlack,f,time

# =============================================================================
# Input:
#   sample = audio clip
#   fs = sampling frequency
#   N = frame size
# Output:
#   arg1 = vector of size nframes of spectral centroids
#   arg2 = vector of size nframes of spectral spreads
#   arg3 = length of output vectors = nframes
# =============================================================================
def spectralCentroid(sample,fs,N):
    nframes = int(len(sample)/N)
    SP = np.zeros([nframes,int(N/2)])
    # Generate Blackman window
    wBlack = scipy.signal.get_window('blackman',N-1,False)
    # Generate a spectrogram matrix
    for n in range(nframes):
        f,t,sp = np.transpose(scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,wBlack,mode='magnitude'))
        SP[n,:] = np.transpose(sp)
    frange = np.shape(sp)[0]
    P = np.zeros([nframes,frange])
    mu = np.zeros(nframes)
    spread = np.zeros(nframes)
    # Loop over FFT frames
    for n in range(nframes):
        # calculate sum of all frequency components in this frame
        sfreq = sum(SP[n,0:frange])
        for k in range(frange):
            # calculate the relative 'probability' of each frequency bin
            P[n,k] = SP[n,k]/sfreq
            # calculate spectral centroid (center of mass) of each frame
            mu[n] = mu[n] + (k*fs/N)*P[n,k]
            # calculate spectral spread of each frame
            spread[n] = np.sqrt(spread[n] + P[n,k]*(k-mu[n])**2)
    return mu,spread,P,nframes

# =============================================================================
# Input:
#   sample = audio clip
#   fs = sampling frequency
#   N = frame size
# Output:
#   arg1 = vector of length nframes of spectral flatness
#   arg3 = length of output vectors = nframes
# =============================================================================
def flatness(sample,fs,N):
    nframes = int(len(sample)/N)
    K = int(N/2)
    SP = np.zeros([nframes,int(N/2)])
    wBlack = scipy.signal.get_window('blackman',N-1,False)
    geo = np.zeros(nframes)
    arith = np.zeros(nframes)
    SF = np.zeros(nframes)
    for n in range(nframes):
        f,t,sp = np.transpose(scipy.signal.spectrogram(sample[n*N:n*N+(N-1)],fs,wBlack,mode='magnitude'))
        SP[n,:] = np.transpose(sp)
        geo[n] = scipy.stats.mstats.gmean(SP[n,:])
        arith[n] = np.mean(SP[n,:])
    SF = geo/arith
    return SF,nframes

# =============================================================================
# Input:
#   P = probability matrix of dimension [nframes,N/2]
# Output:
#   arg1 = vector of length nframes of spectral flux
# =============================================================================
def flux(P):
    nframes = np.shape(P)[0]
    frange = np.shape(P)[1]
    F = np.zeros(nframes)
    for n in range(nframes):
        F[n] = sum((P[n,:]-P[n-1,:])**2)
    return F

def mfcc(SP,f,n,fs,nBanks):
    mel_min = 1127.01048*math.log(1 + 20/700)
    omega_min = (math.e**(mel_min/1127.01048) - 1) * 700
    mel_max = 1127.01048 * math.log(1 + 0.5*fs/700)
    omega_max = (math.e**(mel_max/1127.01048) - 1) * 700
    # Generate frequency array in Hz
    linfreq = np.arange(np.round(omega_min,0),np.round(omega_max,0)+1,1)
    # Map frequency array to mel array
    melfreq = 1127.01048*np.log(1+linfreq/700)
    # Generate nbanks+2 mel frequencies uniformly spaced between mel_min and mel_max, inclusive
    mel = np.linspace(mel_min,mel_max,nBanks+2)
    # Array of center frequencies of each filter
    omega = np.zeros(nBanks+2,np.float64)
    # Populate array
    for i in range(nBanks+2):
        idx = np.argmin(np.abs(melfreq-mel[i]))
        omega[i] = linfreq[idx]
    # Generate filters
    nframes = np.shape(SP)[1]
    frange = np.shape(SP)[0]
    mfcc = np.zeros([nBanks,frange])
    h = np.zeros([nBanks,frange])
    plt.figure(figsize=(8,4),dpi=170)
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency (dB)')
    plt.title('Mel Filter Banks')
    for p in range(nBanks):
        omegaC = omega[p]
        omegaL = omega[p-1]
        omegaR = omega[p+1]
        for k in range(frange):
            freq = f[k]
            if (freq >= omegaL and freq < omegaC):
                h[p,k] = (2/(omegaR-omegaL)) * ((freq-omegaL)/(omegaC-omegaL))
            elif (freq >= omegaC and freq < omegaR):
                h[p,k] = (2/(omegaR-omegaL)) * ((omegaR-freq)/(omegaR-omegaC))
            else:
                pass
        plt.plot(range(len(f)),h[p])
    plt.savefig('figs/melBanks')
    #mfcc[p-1,:] = np.sum((np.abs(h*SP[:,n]))**2)
    return mfcc,h

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

    # Save figures
    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(range(Nloudness),sigma)
    plt.xlabel('FFT Frame')
    plt.ylabel('Loudness')
    plt.title('Loudness: '+filename)
    plt.savefig('figs/loudness_'+filename)
    plt.close()

    # Calculate zero-crossing-rate
    Z,Nzcr = zcr(songSample,512)

    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(range(Nzcr),Z)
    plt.xlabel('FFT Frame')
    plt.ylabel('ZCR')
    plt.title('ZCR: '+filename)
    plt.savefig('figs/zcr_'+filename)
    plt.close()

    # Calculate spectrograms for Hann and Blackman windows
    spectroHann,spectroBlack,freqAxis,timeAxis = spectrogram(songSample,fs,512)
    spectroHann = np.transpose(spectroHann)
    spectroBlack = np.transpose(spectroBlack)

    mel = mfcc(spectroBlack,freqAxis,1,fs,40)

    # save figures
    plt.figure(figsize=(8,4),dpi=170)
    plt.imshow(spectroHann, cmap='inferno')
    plt.gca().invert_yaxis()
    plt.colorbar(shrink=0.5).set_label('Magnitude (dB)')
    plt.title('Hann Window Spectrogram: '+filename)
    plt.xlabel('FFT Frame')
    plt.ylabel('Frequency (bins)')
    plt.savefig('figs/powerHann_'+filename)
    plt.close()

    plt.figure(figsize=(8,4),dpi=170)
    plt.imshow(spectroBlack, cmap='inferno')
    plt.gca().invert_yaxis()
    plt.colorbar(shrink=0.5).set_label('Magnitude (dB)')
    plt.title('Blackman Window Spectrogram: '+filename)
    plt.xlabel('FFT Frame')
    plt.ylabel('Frequency (bins)')
    plt.savefig('figs/powerBlack_'+filename)
    plt.close()

    # calculate statistical vectors, centroid and spread
    centroid,spread,P,nFrames = spectralCentroid(songSample,fs,512)

    # save figures
    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(centroid)
    plt.title('Spectral Centroid: '+filename)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('FFT Frame')
    plt.savefig('figs/centroid_'+filename)
    plt.close()

    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(spread)
    plt.title('Spectral Spread: '+filename)
    plt.ylabel('Spread')
    plt.xlabel('FFT Frame')
    plt.savefig('figs/spread_'+filename)
    plt.close()

    # calculate spectral flatness
    flat,nFrames = flatness(songSample,fs,512)

    # save figure
    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(flat)
    plt.ylabel('Flatness')
    plt.xlabel('FFT Frame')
    plt.title('Spectral Flatness: '+filename)
    plt.savefig('figs/flatness_'+filename)
    plt.close()

    sflux = flux(P)

    plt.figure(figsize=(8,4),dpi=170)
    plt.plot(sflux)
    plt.ylabel('Flux')
    plt.xlabel('FFT Frame')
    plt.title('Spectral Flux: '+filename)
    plt.savefig('figs/flux_'+filename)
    plt.close()

    print('done')
    



