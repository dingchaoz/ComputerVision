import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import os
import cv2
import pandas as pd
import h5py
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

os.chdir('/Users/ejlq/Documents/dingchao/ComputerVision')


def openImg(file = 'croppsroi2017-05-04 12:18:48.695093.png'):
    img = Image.open(file)    
    return img

def resize(img,height = 224,width = 224):
    return img.resize((height,width))
    
def im2Array(img):
    try:
        arr = numpy.array(img.getdata(),numpy.uint8).reshape(img.size[1], img.size[0], 3)
    except:
        grayscaled_arr = numpy.array(img.getdata(),numpy.uint8).reshape(img.size[1], img.size[0])
        arr = cv2.cvtColor(grayscaled_arr, cv2.COLOR_GRAY2RGB)
    #print (arr.shape)
    return arr

def showArray(array):
    plt.imshow(array)

    
def loadImg2Array(file ='roi2017-05-04 12:18:48.695093.png'):
    #print (file)
    
    img = openImg(file)
    resized_img = resize(img)
    arr = im2Array(resized_img)
    return arr

"""
return array in shape(num_img,img_height,img_width,channel)

"""
def loadBatch2Array(directory = 'cropps/'):
    arrays = [loadImg2Array(directory+name) for name in os.listdir(directory)[1:]]
    np_array = np.array(arrays, dtype=np.float64)
    return np_array