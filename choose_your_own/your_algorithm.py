#!/usr/bin/python

import sys
from time import time
sys.path.append("../tools/")

#required imports
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as numpy

##training
neighbors = ['auto', 'ball_tree', 'kd_tree', 'brute']
adaboosts = ['SAMME.R', 'SAMME']
classifiers = ['RandomForest', 'AdaBoost', 'K-NearestNeighbor']
classifierType = classifiers[2]

if classifierType == 'RandomForest':
    clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=1)
elif classifierType == 'AdaBoost':
    clf = AdaBoostClassifier(algorithm=adaboosts[0], n_estimators=50, learning_rate=1.0)
elif classifierType == 'K-NearestNeighbor':
    clf = KNeighborsClassifier(algorithm=neighbors[0], n_neighbors=5, leaf_size=30)
else:
    clf = GaussianNB()

print "classifier:", clf
t0 = time()
print "training classifier..."
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

##predictions
t0 = time()
print "making predictions..."
predictions = clf.predict(features_test)
print "predictions time:", round(time()-t0, 3), "s"

##accuracy
accuracy = accuracy_score(predictions, labels_test)
print "accuracy: ", accuracy

### visualization code (prettyPicture) to show you the decision boundary
fileName = classifierType + ".png"
try:
    prettyPicture(clf, features_test, labels_test, fileName)
except NameError as e:
    print e
