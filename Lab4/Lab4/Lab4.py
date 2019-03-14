import lab04_funcs as lf
from tkinter import filedialog
import matplotlib.pyplot as plt

d_taps = lf.load_d_taps('the_d_taps.txt')

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
# Loop over files selected
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]
    filepath = f

    # calculate sub-band filter coefficients
    coefficients = lf.pqmf(f)

    # calculate reconstruction coefficients
    print('Reconstructing ...\n')
    recons = lf.ipqmf(coefficients)
    
    # plot reconstruction coefficients
    plt.figure(dpi=170)
    plt.plot(recons, linewidth=0.25)
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.title('Reconstruction coefficients for first 5 seconds of ' + filename + '.wav')
    plt.show()
    plt.savefig('./figures/' + filename + '_rc_filt.png')



print('done')
