#!/usr/local/bin/python3
import numpy as np
from PIL import Image
import sys
import math
from numpy import linalg as LA

# NOTE: x is "column" and y is "row"!!!!!  x and y are floats
# this code processes all 3 color planes at once
# the input picture is a numpy 3-D array of size (R, C, 3)

def bilinear_interp(x, y, pic) :
  R = pic.shape[0] # image rows
  C = pic.shape[1] # image cols

  # Below we return "black" immediately if we aren't guaranteed
  # that both the floor and ceiling exist, i.e., if we would go
  # outside the picture boundaries
  if x <= 0 or x >= C - 1 or y <= 0 or y >= R - 1 :
    retval = np.asarray( [0., 0., 0.] )  
  else:
    # Do the bi-linear interpolation
    xf = math.floor( x )
    xc = math.ceil( x )
    yf = math.floor( y )
    yc = math.ceil( y )
    
    ul = pic[yf, xf] # upper-left pixel
    ur = pic[yf, xc] # upper-right pixel
    ll = pic[yc, xf] # lower-left pixel
    lr = pic[yc, xc] # lower-right pixel

    alpha = x - xf
    val1 = (1 - alpha)*ul + alpha*ur
    val2 = (1 - alpha)*ll + alpha*lr
    alpha = y - yf

    return( (1-alpha)*val1 + alpha*val2 )
