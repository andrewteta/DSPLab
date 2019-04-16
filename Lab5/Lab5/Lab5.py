from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lab_functions as lf

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
for f in files:
    filename = str.split(f,'/')[-1]
    filename = str.split(filename,'.')[0]
    filepath = f
    image = np.asarray(Image.open(f))
    Image.fromarray(image).save('figures/' + filename + '.png')
    loss_factor = 8
    processed = lf.dctmgr(image, loss_factor)
    x = lf.idctmgr(processed, loss_factor)
    Image.fromarray(x.astype(np.uint8)).save('figures/' + filename + '_lf' + str(loss_factor) + '.png')

print('done')