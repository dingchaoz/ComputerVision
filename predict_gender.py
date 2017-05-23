import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import os
import cv2
import pandas as pd
import h5py
import keras
from keras.models import load_model
from keras.utils.np_utils import probas_to_classes
import collections


modelPath = 'weights-improvement-05-0.91.hdf5'
modelAgePath = 'weights-improvement-08-8.15.hdf5'



def openImg(file = 'cropps/roi2017-05-04 12:18:48.695093.png'):
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

    
def loadImg2Array(file ='cropps/roi2017-05-04 12:18:48.695093.png'):
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
    np_array -= np.mean(np_array,axis = 0)
    np_array /= np.std(np_array,axis = 0)
    return np_array

def loadVGG(model = modelPath):
	print ('load model',modelPath)
	model = load_model(model)
	return model

def loadVGGAge(agemodel = modelAgePath):
	print ('load age model',modelAgePath)
	agemodel = load_model(agemodel)
	return agemodel
	

def predict_vgg(model,x_test):
    y_proba = model.predict(x=np.array(x_test))
    y_classes = probas_to_classes(y_proba)
    counter = collections.Counter(y_classes)
    most_common = counter.most_common(1)[0][0]
    if most_common  == 1:
    	gender = 'male'
    elif most_common  == 0:
    	gender = 'female'
    return gender

def predict_age_vgg(agemodel,x_test):
    age = agemodel.predict(x=np.array(x_test))

    return age

