#!/usr/local/bin/python3
import numpy as np
import math
import sys
from PIL import Image

if len( sys.argv ) != 3 :
  print(f"usage is: {sys.argv[0]} image_file_1 image_file_2")
  sys.exit()

# Input the images and print errors as necessary
try:
  imf_1 = Image.open(sys.argv[1])
except IOError:  
  print(f"ERROR: could not open {sys.argv[1]}")
  sys.exit()

try:
  imf_2 = Image.open(sys.argv[2])
except IOError:  
  print(f"ERROR: could not open {sys.argv[2]}")
  sys.exit()

im_1 = np.asarray( imf_1.convert('L'), np.float )
im_2 = np.asarray( imf_2.convert('L'), np.float )

if im_1.shape != im_2.shape :
  print(f"ERROR: images are not same size")
  sys.exit(-1)

print(f"Image rows: {im_1.shape[0]}")
print(f"Image cols: {im_1.shape[1]}")

# denom holds the error
denom = np.sum( (im_1 - im_2)**2 ) / (im_1.shape[0]*im_1.shape[1])

# Compute the PSNR
if denom == 0 :
  print(f"These images are identical")
else:
  PSNR = 10 * np.log10( 255**2 / denom )
  print(f"PSNR: {PSNR} dB")
