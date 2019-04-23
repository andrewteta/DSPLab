#!/usr/local/bin/python3
# For the titanic data set

import numpy as np
import math
import sys
import csv

def get_titanic_all( filename ):
  fd_r = open(filename, 'r')
  datareader = csv.reader(fd_r, dialect='excel')

  # Read the labels from the file, we won't need them
  a = next(datareader)

  # Return class (0), sex (3), age (4), sibsp (5), parch (6), fare(8),
  # embarked (10)
  # 
  # The numbers in parantheses are the corresponding column indices
  # in the CSV table.  Note that the outcome we
  # are trying to predict, "survived", is in column 1

  # The values are encoded into the data matrix to analyze as follows:
  # class: 3 columns. (1st, 2nd, or 3rd class)
  # sex:   1 column.  (M / F)
  # age:   81 cols.  First col indicates age known/unknown, the following
  #        80 cols are all zero except for a 1 at the appropriate age
  # sibsp: 9 cols. 0 to 8 (flags number of siblings and parents on board)
  # parch: 10 cols. 0 to 9 (flags number of parents/children onboard)
  # fare:  1 col. a real number (price)
  # embarked: 3 cols. a 1 to indicate C/S/Q as embarkation point

  X = np.zeros( (0,108), np.float)
  xt = np.zeros(108, np.float)
  y = np.zeros(0, np.float)

  # Parse the data set and build a feature matrix, X, using just
  # the values we care about.  The following table is useful:
  #
  # parameter col_in_raw_data start_col_in_X num_cols
  #   class          0               0          3  
  #   sex            3               3          1          
  #   age            4               4          81
  #   sibsp          5               85         9
  #   parch          6               94         10
  #   fare           8               104        1
  #   embarked       10              105        3

  for ctr, val in enumerate(datareader):
    xt[:] = 0
    
    # Class
    tv = np.int(val[0])
    if tv == 1 : xt[0] = 1
    elif tv == 2 : xt[1] = 1
    else: xt[2] = 1

    # Sex
    if val[3] == 'female': xt[3] = 1;
    
    # Age (some values in the data set are blank!
    if val[4] == '':
      xt[4] = 1.
    else:  
      xt[5 + np.int(np.round(np.float(val[4])))] = 1
    
    # Sibsp
    tv = np.int( val[5] )
    xt[85 + tv] = 1

    # Parch
    tv = np.int( val[6] )
    xt[94 + tv] = 1

    # Fare
    xt[104] = np.float( val[8] )

    # Embarked at
    if val[10] == 'C' : xt[105] = 1
    elif val[10] == 'Q' : xt[106] = 1
    else: xt[107] = 1
    
    # add a row to X and a column to y
    X = np.vstack( [X, xt] )
    y = np.hstack( [y, np.float(val[1]) ] )

  return( X, y )
