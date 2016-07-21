#!/usr/bin/env python3
"""
 This module implements a demo for state farm that will try to classify pics of cats and dogs.
"""
import glob
import PIL
import sklearn
import sklearn.svm
import numpy as np
import json
from flask import request,Flask


IMAGE_SIZE(250,250)

def training_data(path,count =100):
	""" return a 2D numpy array of at most "count" flattened images from given global path"""
	
	file_list = glob.glob(path)[:count]
	
	images = []
	for fpath in file_list:
		img = PIL.Image.open(fpath).resize(IMAGE_SIZE)
		arr = np.array(img).flatten()
		images.append(arr)
		
	final_arr = np.vstack(images)
	return final_arr
	
def make_svm_model(train,test,n_components):
	pca = sklearn.decomposition.PCA(components = n_components)
	clf = sklearn.svm.SVC()
	clf.fit(train,feat)
	
	return clf,pca
	
def predict(test_data,clf,pca):

	
train_arr_dogs = training_data("../training/dog/")
train_arr_cats = training_data("../training/cat/")

train_arr = np.vstack(train_arr_dogs,train_arr_cats)
feat_arr_dogs = np.zeros(train_arr_dogs.shape[0])
feat_arr_cats = np.zeros(train_arr_cats.shape[0])
feat_arr = np.hstack(feat_arr_dogs,feat_arr_cats)

@app.route("/predict"/)
def predict():
	img = PIL.Image.open().re
	test1_arr
	prediction = 
	return json.dumps("prediction":
	
clf,pca = make_svm_model(train_arr,feat_arr,.9)
test1 = PIL.Image.open("../test/1.jgp").resize(IMAGE_SIZE)
test1_arr = pca.transform(np.array(test1).flatten())

print(clf.predict(test1_arr))

if __name__ = "_main_"
	

def main():


