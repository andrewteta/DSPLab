from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lab_functions as lf

# UI dialog to select files -> selection of multiple files will run all functions for each file
files = filedialog.askopenfilenames()
for f in files:
    image = np.asarray(Image.open(f))
    processed = lf.dctmgr(image)

print('done')