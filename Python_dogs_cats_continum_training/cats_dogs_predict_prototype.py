# commit
hg init

conda create -n state_farm python-3
source deactive state_farm
conda install ipython

#!/usr/bin/env python3
"""
 This module implements a demo for state farm that will try to classify pics of cats and dogs.
"""

import PIL
import sklearn
import sklearn.svm
import numpy as np

## The followings are tests before starting real coding, REMOVE them for clean coding
IMAGE_SIZE = (250,250)

dog1 = PIL.Image.open("../train/dogs/dog1.jpg").resize(IMAGE_SIZE)
cat1 = 

dogarr1 = np.array(dog1).flatten()
catarr1 = np.array(cat1).flatten()

# Note that 

train_arr = np.vstack(dogarr1,catarr1)
feat_arr = np.array([0,1])

clf = sklearn.svm.SVC()
clf.fit(train_arr,feat_err)

test1 = PIL.Image.open("../test/1.jpg").resize(IMAGE_SIZE)
testarr1


print(clf.predict(testarr1))