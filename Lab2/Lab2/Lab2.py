import numpy as np
from PIL import Image

def grayscale(image):
    imageIn = np.asarray(pil_imageIn,np.uint8)
    rows = imageIn.shape[0]
    cols = imageIn.shape[1]

    # convert image to greyscale
    Y = [0.299,0.587,0.114]
    imGray = np.dot(imageIn,Y)
    imGray = imGray.astype(np.uint8)

    # convert ndarray into linear array
    linIm = np.reshape(imGray,rows*cols)

    # clip out of bounds values
    linIm = np.where(linIm < 0,0,linIm)
    linIm = np.where(linIm > 255,255,linIm)

    # reshape back into 2D array
    imGray = np.reshape(linIm,[rows,cols])

    return imGray

pil_imageIn = Image.open('images/test01.jpg')

gray1 = grayscale(pil_imageIn)

# Output image to file
Image.fromarray(gray1).save('figs/test01_out.jpg')