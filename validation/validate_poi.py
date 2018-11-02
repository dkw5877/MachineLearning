#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
sort_keys = '../tools/python2_lesson13_keys.pkl'

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)

# if using python 3 we need to sort the keys
# data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#test_size=1 for all data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

##training
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

##predictions
t0 = time()
predictions = clf.predict(features)
print "prediction time:", round(time()-t0, 3), "s"

##accuracy
## quiz1 expected answer =  0.9894736842105263
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions, labels)
print "accuracy_score = ", accuracy

#accuracy of the classifier
#quiz 2 expected answer = 0.7241379310344828
acc = clf.score(features_test, labels_test)
print "clf score = ", acc


