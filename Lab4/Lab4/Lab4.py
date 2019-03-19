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
    print(filename + f' error = {error}')

    # plot delay
    plt.figure(dpi=170)
    plt.plot(data[0:1000], linewidth=0.35)
    plt.plot(recons[0:1000], linewidth=0.35)
    if (filename == 'cast'):
        plt.clf()
        plt.plot(data[2850:3850], linewidth=0.35)
        plt.plot(recons[2850:3850], linewidth=0.35)
    plt.title('Original waveform vs. reconstructed')
    plt.savefig('./figures/synthesis/' + filename + '_delay.png')
    plt.close()
    
    #calculate error when leaving out high frequency sub-bands
    #errors = np.zeros(32)
    #for band in range(31, 0, -1):
    #    thebands[band:32] = 0
    #    coefficients = np.reshape(coefficients, (32, -1)).T
    #    selective_output = lf.ipqmf(coefficients, thebands)
    #    selective_output = selective_output.flatten()
    #    # calculate error
    #    r = selective_output[delay:delay + 5000]
    #    d = data[0:5000]
    #    errors[band] = np.amax(r - d)
    #plt.figure(dpi=170)
    #plt.bar(range(32), errors)
    #plt.title('Error as a function of high frequency bands excluded')
    #plt.savefig('./figures/synthesis/' + filename + '_select.png')

    thebands = np.ones(32)
    coefficients = np.reshape(coefficients, (32, -1)).T
    if filename == 'gilberto':
        thebands[16:32] = 0
    elif filename == 'sine1':
        thebands[0] = 0
        thebands[2:32] = 0
    elif filename == 'sine2':
        thebands[1:32] = 0
    elif filename == 'handel':
        thebands[8:32] = 0
    elif filename == 'sample1':
        thebands[8:32] = 0
    elif filename == 'sample2':
        thebands[15:32] = 0
    selective_output = lf.ipqmf(coefficients, thebands)
    selective_output = selective_output.flatten()
    # calculate error
    r = selective_output[delay:delay + 5000]
    d = data[0:5000]
    error = np.amax(r - d)
    print(filename + f' selective error = {error}')


print('done')
