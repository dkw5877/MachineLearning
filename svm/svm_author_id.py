#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

### reduce data set size
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

###train the classifier using Support Vector Classification (SVC)
#set gamma for rbf to avoid warning 

t0 = time()
classifier = svm.SVC(kernel='rbf', gamma='scale', C=100)
classifier.fit(features_train, labels_train)
print "training time = ", round(time() -  t0, 3), "s"

##make predictions
t0 = time()
predictions = classifier.predict(features_test)
print "prediction time = ", round(time() - t0, 3), "s"

answer1 = predictions[9]
print "10th = ", answer1

answer2 = predictions[25]
print "26th = ", answer2

answer3 = predictions[49]
print "50th = ", answer3


##accuracy
accuracy = accuracy_score(labels_test, predictions)
print "accuracy score: ", accuracy

#########################################################


