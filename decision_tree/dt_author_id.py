#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from email_preprocess import preprocess
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### Visualization on terrain data
#features_train, labels_train, features_test, labels_test = makeTerrainData()


#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

#features
numcolumns = len(features_train[0])
print "columns = ", numcolumns

##training
classifier = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

##predictions
t0 = time()
predictions = classifier.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

##accuracy
accuracy = accuracy_score(predictions, labels_test)
print "accuracy = ", accuracy

#########################################################
## NOTE: Visualizztion only works for Terrain Data
try:
    prettyPicture(classifier, features_test, labels_test)
except NameError as e:
    print (e)