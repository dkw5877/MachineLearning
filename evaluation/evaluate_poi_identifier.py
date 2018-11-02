#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import numpy as np
import pickle
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

##training
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

##predictions
t0 = time()
predictions = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

#number of POIs in the test set = 4
#quiz 28 accepts 3 or 4 as answer
print "POIs in the test set: ", int( sum(predictions) )

#how many people total in test set
print "Total people in test set ", len(features_test)

#accuracy of biased identifier
#quiz answers = 0.862 or 0.896
pois = int( sum(predictions))
num_people = len(features_test)
numerator = num_people - pois
print "accuracy of biased identifier: ",  numerator/float(len(features_test))

#Number of True Positives
#quiz answer is zero, not sure how this is arrived at programmatically
from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(predictions, labels)

#precision score 
#quiz answer is 0.0
from sklearn.metrics import precision_score
score = precision_score(labels_test, predictions) 
print "precision score: ", score

#recall score 
#quiz answer is 0.0
from sklearn.metrics import recall_score
score = recall_score(labels_test, predictions) 
print "recall score: ", score


## base on sample data from quiz 

#How Many True Positives
# where prediction (1) matches label (1)
# quiz answer = 6


#How Many True Negatives?
# where prediction (0 matches label (0)
# quiz answer = 9


#How many False Positives
# where prediction (1) matches label (0)
# quiz answer = 3

#How many false negatives 
# where prediction (0) matches label (1)
# quiz answer = 2

#Precision of classifier
#true_positives / (true_positives + false_positives)
# 6 / (6 + 3) = 0.666

#Recall of this classifier
#true_positives / (true_positives + false_negatives)
# 6 / (6 + 2) = 0.75

##accuracy
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(predictions, labels)
# print "accuracy_score = ", accuracy

#accuracy of the classifier
# acc = clf.score(features_test, labels_test)
# print "clf accuracy = ", acc







