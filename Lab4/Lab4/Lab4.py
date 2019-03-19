import lab04_funcs as lf
from tkinter import filedialog
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

d_taps = lf.load_d_taps('the_d_taps.txt')

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
# Loop over files selected
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]
    filepath = f

    # read .wav file into numpy array and extract sampling frequency
    fs, data = scipy.io.wavfile.read(filepath)
    
    # extract first 5 seconds of .wav file
    data = data[0 : int(5 * fs)]

    # plot input samples
    plt.figure(dpi=170)
    plt.plot(data, linewidth=0.25)
    plt.title(filename + '.wav')
    plt.savefig('./figures/' + filename + '.png')
    plt.close()

    # calculate sub-band filter coefficients
    print('Analyzing ' + filename + ' ... \n')
    coefficients = lf.pqmf(data)

    # vector to filter subbands
    thebands = np.ones(32)
    #for i in range(32):
    #    if (i % 2):
    #        thebands[i] = 0

    # calculate reconstruction coefficients
    print('Reconstructing ' + filename + ' ...\n')
    recons = lf.ipqmf(coefficients, thebands)

    # transpose and flatten to sort data into incrementing groups of subband coefficients
    coefficients = coefficients.T.flatten()

    # plot sub-band coefficients
    plt.figure(dpi=170)
    plt.plot(coefficients, linewidth=0.25)
    xlocs = np.asarray(range(0, len(coefficients), int(len(coefficients)/10)))
    xvals = np.asarray(xlocs/np.amax(range(len(coefficients)))*fs/2, int)
    plt.xticks(xlocs, xvals, rotation=20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Sub-band coefficients for first 5 seconds of ' + filename + '.wav')
    plt.tight_layout()
    plt.savefig('./figures/analysis/' + filename + '.png')
    plt.close()

    recons = recons.flatten()
    
    # plot reconstruction coefficients
    plt.figure(dpi=170)
    plt.plot(recons, linewidth=0.25)
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.title('Reconstruction of first 5 seconds of ' + filename + '.wav')
    #plt.show()
    plt.savefig('./figures/synthesis/' + filename + '.png')
    plt.close()

    delay = 512 - 31;

    # calculate error
    r = recons[delay:1000 + delay]
    d = data[0:1000]
    error = np.amax(r - d)

    # plot delay
    plt.figure(dpi=170)
    plt.plot(data[0:1000], linewidth=0.35)
    plt.plot(recons[0:1000], linewidth=0.35)
    if (filename == 'cast'):
        plt.clf()
        plt.plot(data[2850:3850], linewidth=0.35)
        plt.plot(recons[2850:3850], linewidth=0.35)
    plt.title('Original waveform vs. reconstructed: error = %.2f' %error)
    plt.savefig('./figures/synthesis/' + filename + '_delay.png')
    plt.close()



print('done')
