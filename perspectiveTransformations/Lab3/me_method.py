#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import math

# block_bw: black/white block from frame n+1 (will search for a 
#           match for this block in Frame n)
# im_bw, im_rgb: Frame n in black/white and rgb numpy arrays
# row_start: image row where search in Frame n will start
# col_start: image col where search in Frame n will start
# search_range: +/- pixel range over which a match will be sought
# RETURN VALUES: a prediction for "block" (in color), the best mse,
#                the row/col offsets of the best match (positive offsets
#                are down and to the right)
# NOTE: search range is automatically restricted to ensure that we don't
#       search outside the boundaries of frame n

def motion_match(row_start, col_start, search_range, block_bw, im_bw, im_rgb):
  # row 0 is the top row in the image
  # col 0 is the left most column in the image
  rows = im_bw.shape[0]
  cols = im_bw.shape[1]
  rt = row_start - search_range  # row top
  rb = row_start + search_range  # row bottom
  cl = col_start - search_range  # col left
  cr = col_start + search_range  # col right

  # Constrain search to stay within Frame n boundaries
  if rt < 0 : rt = 0
  if rb > rows - 16 : rb = rows - 16
  if cl < 0 : cl = 0
  if cr > cols - 16 : cr = cols - 16

  # Declare array to hold the search results and do the search
  mse = np.zeros( (rb - rt + 1, cr - cl + 1), np.float )
  for k in range(rt, rb + 1) :
    for l in range(cl, cr + 1) :
      mse[k-rt, l-cl] = np.mean(  (  block_bw.astype(np.float)
                                   - im_bw[k:k+16, l:l+16].astype(np.float)  
                                  )**2   
                               )
  # min_index is w.r.t flattened array.  Find the index of the minimum
  # mse, and convert it to 2D indices
  min_index = np.argmin( mse )
  min_row = min_index // mse.shape[1]
  min_col = min_index % mse.shape[1]

  # extract the prediction for the block in rgb format, and return
  # relevant information about the best match's offset and mse
  pred_block = im_rgb[rt+min_row:rt+min_row+16, cl+min_col:cl+min_col+16]
  return pred_block, mse[min_row, min_col], \
         rt + min_row - row_start, cl + min_col - col_start

def motion_match_mae(row_start, col_start, search_range, block_bw, im_bw, im_rgb):
  # row 0 is the top row in the image
  # col 0 is the left most column in the image
  rows = im_bw.shape[0]
  cols = im_bw.shape[1]
  rt = row_start - search_range  # row top
  rb = row_start + search_range  # row bottom
  cl = col_start - search_range  # col left
  cr = col_start + search_range  # col right

  # Constrain search to stay within Frame n boundaries
  if rt < 0 : rt = 0
  if rb > rows - 16 : rb = rows - 16
  if cl < 0 : cl = 0
  if cr > cols - 16 : cr = cols - 16

  # Declare array to hold the search results and do the search
  mse = np.zeros( (rb - rt + 1, cr - cl + 1), np.float )
  for k in range(rt, rb + 1) :
    for l in range(cl, cr + 1) :
      mse[k-rt, l-cl] = np.mean(  np.abs(  block_bw.astype(np.float)
                                   - im_bw[k:k+16, l:l+16].astype(np.float)  
                                  )
                               )
  # min_index is w.r.t flattened array.  Find the index of the minimum
  # mse, and convert it to 2D indices
  min_index = np.argmin( mse )
  min_row = min_index // mse.shape[1]
  min_col = min_index % mse.shape[1]

  # extract the prediction for the block in rgb format, and return
  # relevant information about the best match's offset and mse
  pred_block = im_rgb[rt+min_row:rt+min_row+16, cl+min_col:cl+min_col+16]
  return pred_block, mse[min_row, min_col], \
         rt + min_row - row_start, cl + min_col - col_start