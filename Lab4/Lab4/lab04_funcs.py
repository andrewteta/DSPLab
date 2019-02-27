#!/usr/local/bin/python3
import numpy as np
import math

def load_c_taps( filename_path ):
  fd = open(filename_path, 'r')
  c = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    c[ctr] = np.float( value )
  fd.close()
  return c

def load_d_taps( filename_path ):
  fd = open(filename_path, 'r')
  d = np.zeros(512, np.float)
  for ctr, value in enumerate(fd):
    d[ctr] = np.float( value )
  fd.close()
  return d
